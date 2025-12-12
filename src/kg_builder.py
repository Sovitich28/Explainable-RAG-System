"""
Knowledge Graph Builder Module
Extracts entities and relationships from text chunks and populates Neo4j.
"""

import json
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv  # Added import
from groq import Groq
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()  # Added function call


class KnowledgeGraphBuilder:
    """Build and populate the Knowledge Graph in Neo4j."""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize the KG builder.

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Configure Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    def create_schema(self):
        """Create the Knowledge Graph schema with constraints and indexes."""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT policy_name IF NOT EXISTS FOR (p:Policy) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT target_name IF NOT EXISTS FOR (t:Target) REQUIRE t.name IS UNIQUE",
                "CREATE CONSTRAINT legislation_name IF NOT EXISTS FOR (l:Legislation) REQUIRE l.name IS UNIQUE",
                "CREATE CONSTRAINT country_iso IF NOT EXISTS FOR (c:Country) REQUIRE c.iso_code IS UNIQUE",
                "CREATE CONSTRAINT renewable_name IF NOT EXISTS FOR (r:RenewableSource) REQUIRE r.name IS UNIQUE",
                "CREATE CONSTRAINT sector_name IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Constraint already exists or error: {e}")

            # Create indexes
            indexes = [
                "CREATE INDEX policy_year IF NOT EXISTS FOR (p:Policy) ON (p.year)",
                "CREATE INDEX target_deadline IF NOT EXISTS FOR (t:Target) ON (t.deadline)",
                "CREATE INDEX legislation_year IF NOT EXISTS FOR (l:Legislation) ON (l.year)",
            ]

            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    print(f"Index already exists or error: {e}")

            print("Schema created successfully")

    def extract_entities_and_relations(self, chunk: Dict[str, any]) -> Dict[str, any]:
        """
        Extract entities and relationships from a text chunk using LLM.

        Args:
            chunk: Text chunk with metadata

        Returns:
            Dictionary containing extracted entities and relationships
        """
        prompt = f"""You are an expert in extracting structured information from policy documents.

Extract entities and relationships from the following text about EU environmental policy and renewable energy.

Text: {chunk['text']}

Extract the following types of entities:
1. Policy: Environmental policies (e.g., European Green Deal)
2. Target: Specific renewable energy targets with values and deadlines
3. Legislation: Laws and directives (e.g., RED III)
4. Country: EU member states mentioned
5. RenewableSource: Types of renewable energy (solar, wind, etc.)
6. Sector: Economic sectors (transport, industry, buildings)

Extract relationships between entities:
- MANDATES (Policy → Target)
- IMPLEMENTS (Legislation → Policy)
- APPLIES_TO (Target → Country)
- PROMOTES (Policy → RenewableSource)
- IMPACTS (Policy → Sector)
- SET_BY (Target → Legislation)
- SUPPORTS (Country → RenewableSource)

Return the result as a JSON object with this structure:
{{
  "entities": {{
    "Policy": [list of policies with name, description, year],
    "Target": [list of targets with name, value, unit, deadline, description],
    "Legislation": [list of legislation with name, type, year, description],
    "Country": [list of countries with name, iso_code],
    "RenewableSource": [list of renewable sources with name, type],
    "Sector": [list of sectors with name, description]
  }},
  "relationships": [
    {{"type": "MANDATES", "from": {{"label": "Policy", "name": "..."}}, "to": {{"label": "Target", "name": "..."}}, "properties": {{}}}},
    ...
  ]
}}

Only extract information explicitly mentioned in the text. Be precise and factual.
"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert knowledge graph extraction system. Output valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            result_text = chat_completion.choices[0].message.content
            # Extract JSON from response
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            json_str = result_text[json_start:json_end]

            extracted = json.loads(json_str)
            return extracted

        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {"entities": {}, "relationships": []}

    def populate_graph(self, chunk: Dict[str, any], extracted_data: Dict[str, any]):
        """
        Populate the Neo4j graph with extracted entities and relationships.

        Args:
            chunk: Original text chunk with metadata
            extracted_data: Extracted entities and relationships
        """
        with self.driver.session() as session:
            # Add entities
            entities = extracted_data.get("entities", {})

            # Add Policies
            for policy in entities.get("Policy", []):
                if not policy.get("name"):
                    continue
                session.run(
                    """
                    MERGE (p:Policy {name: $name})
                    SET p.description = $description,
                        p.year = $year,
                        p.source_document = $source_document,
                        p.source_chunk_id = $chunk_id
                """,
                    name=policy.get("name"),
                    description=policy.get("description", ""),
                    year=policy.get("year", 0),
                    source_document=chunk["source_document"],
                    chunk_id=chunk["chunk_id"],
                )

            # Add Targets
            for target in entities.get("Target", []):
                if not target.get("name"):
                    continue
                session.run(
                    """
                    MERGE (t:Target {name: $name})
                    SET t.value = $value,
                        t.unit = $unit,
                        t.deadline = $deadline,
                        t.description = $description,
                        t.source_document = $source_document,
                        t.source_chunk_id = $chunk_id
                """,
                    name=target.get("name"),
                    value=target.get("value", ""),
                    unit=target.get("unit", ""),
                    deadline=target.get("deadline", 0),
                    description=target.get("description", ""),
                    source_document=chunk["source_document"],
                    chunk_id=chunk["chunk_id"],
                )

            # Add Legislation
            for legislation in entities.get("Legislation", []):
                if not legislation.get("name"):
                    continue
                session.run(
                    """
                    MERGE (l:Legislation {name: $name})
                    SET l.type = $type,
                        l.year = $year,
                        l.description = $description,
                        l.source_document = $source_document,
                        l.source_chunk_id = $chunk_id
                """,
                    name=legislation.get("name"),
                    type=legislation.get("type", ""),
                    year=legislation.get("year", 0),
                    description=legislation.get("description", ""),
                    source_document=chunk["source_document"],
                    chunk_id=chunk["chunk_id"],
                )

            # Add Countries
            for country in entities.get("Country", []):
                iso_code = country.get("iso_code")
                if not iso_code and country.get("name"):
                    iso_code = country.get("name")[:2].upper()

                if not iso_code:
                    continue

                session.run(
                    """
                    MERGE (c:Country {iso_code: $iso_code})
                    SET c.name = $name
                """,
                    iso_code=iso_code,
                    name=country.get("name", ""),
                )

            # Add Renewable Sources
            for renewable in entities.get("RenewableSource", []):
                if not renewable.get("name"):
                    continue
                session.run(
                    """
                    MERGE (r:RenewableSource {name: $name})
                    SET r.type = $type
                """,
                    name=renewable.get("name"),
                    type=renewable.get("type", ""),
                )

            # Add Sectors
            for sector in entities.get("Sector", []):
                if not sector.get("name"):
                    continue
                session.run(
                    """
                    MERGE (s:Sector {name: $name})
                    SET s.description = $description
                """,
                    name=sector.get("name"),
                    description=sector.get("description", ""),
                )

            # Add relationships
            for rel in extracted_data.get("relationships", []):
                rel_type = rel.get("type")
                from_data = rel.get("from", {})
                to_data = rel.get("to", {})

                if not (rel_type and from_data.get("name") and to_data.get("name")):
                    continue

                from_label = from_data.get("label")
                from_name = from_data.get("name")
                to_label = to_data.get("label")
                to_name = to_data.get("name")
                properties = rel.get("properties", {})

                # Construct Cypher query for relationship
                # Note: We match by name for most nodes, but Country uses iso_code.
                # This simple builder assumes name is the key for everything in relationships.
                # If Country is involved, we might need to adjust, but let's assume the LLM returns names consistent with nodes.

                query = f"""
                    MATCH (from:{from_label}) WHERE from.name = $from_name OR from.iso_code = $from_name
                    MATCH (to:{to_label}) WHERE to.name = $to_name OR to.iso_code = $to_name
                    MERGE (from)-[r:{rel_type}]->(to)
                """

                # Add properties to relationship
                if properties:
                    set_clause = ", ".join(
                        [f"r.{key} = ${key}" for key in properties.keys()]
                    )
                    query += f" SET {set_clause}"

                try:
                    session.run(
                        query, from_name=from_name, to_name=to_name, **properties
                    )
                except Exception as e:
                    print(f"Error creating relationship {rel_type}: {e}")

    def build_from_chunks(self, chunks: List[Dict[str, any]]):
        """
        Build the complete knowledge graph from text chunks.

        Args:
            chunks: List of text chunks with metadata
        """
        print(f"Building knowledge graph from {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}: {chunk['chunk_id']}")

            # Extract entities and relationships
            extracted = self.extract_entities_and_relations(chunk)

            # Populate graph
            self.populate_graph(chunk, extracted)

        print("Knowledge graph build complete!")

    def get_graph_stats(self) -> Dict[str, int]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with node and relationship counts
        """
        with self.driver.session() as session:
            stats = {}

            # Count nodes by label
            labels = [
                "Policy",
                "Target",
                "Legislation",
                "Country",
                "RenewableSource",
                "Sector",
            ]
            for label in labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                stats[label] = result.single()["count"]

            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats["Total_Relationships"] = result.single()["count"]

            return stats


def main():
    """Main execution function."""
    # Neo4j connection details
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # Initialize builder
    kg_builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Create schema
    kg_builder.create_schema()

    # Load chunks (assuming they were created by data_preparation.py)
    from data_preparation import main as prepare_data

    chunks = prepare_data()

    # Build graph (limit to first 3 chunks for demo to save API costs)
    kg_builder.build_from_chunks(chunks[:3])

    # Get statistics
    stats = kg_builder.get_graph_stats()
    print("\nKnowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    kg_builder.close()


if __name__ == "__main__":
    main()
