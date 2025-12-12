"""
RAG System Module
Implements hybrid retrieval (vector + graph) and explainable generation.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
from groq import Groq
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


class ExplainableRAGSystem:
    """Explainable RAG system with hybrid retrieval."""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize the RAG system.

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        # Neo4j connection
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )

        # Configure Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)

        # Embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # LlamaIndex setup
        Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        Settings.llm = None

        # Vector index (will be populated)
        self.vector_index = None
        self.chunks = []

    def close(self):
        """Close connections."""
        self.neo4j_driver.close()

    def load_and_index_chunks(self, chunks: List[Dict[str, any]]):
        """
        Load text chunks and create vector index.

        Args:
            chunks: List of text chunks with metadata
        """
        self.chunks = chunks

        # Create LlamaIndex documents
        documents = []
        for chunk in chunks:
            doc = Document(
                text=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "source_document": chunk["source_document"],
                    "header": chunk.get("header", ""),
                },
            )
            documents.append(doc)

        # Create vector index
        self.vector_index = VectorStoreIndex.from_documents(documents)
        print(f"Indexed {len(documents)} chunks in vector store")

    def vector_search(self, query: str, top_k: int = 3) -> List[Dict[str, any]]:
        """
        Perform vector similarity search.

        Args:
            query: User query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        if not self.vector_index:
            return []

        # Query the index
        retriever = self.vector_index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        # Extract source nodes
        results = []
        for node in nodes:
            results.append(
                {
                    "text": node.node.text,
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
            )
        return results
        return results

    def graph_search(self, query: str) -> Dict[str, any]:
        """
        Perform graph-based search using Neo4j.

        Args:
            query: User query

        Returns:
            Relevant subgraph with nodes and relationships
        """
        # Convert query to graph query using LLM
        cypher_query = self._generate_cypher_query(query)

        if not cypher_query:
            return {"nodes": [], "relationships": []}

        # Execute Cypher query
        with self.neo4j_driver.session() as session:
            try:
                result = session.run(cypher_query)

                nodes = []
                relationships = []

                for record in result:
                    # Extract nodes and relationships from result
                    for key in record.keys():
                        value = record[key]
                        if hasattr(value, "labels"):  # It's a node
                            node_dict = dict(value)
                            node_dict["labels"] = list(value.labels)
                            node_dict["id"] = value.id
                            nodes.append(node_dict)
                        elif hasattr(value, "type"):  # It's a relationship
                            rel_dict = dict(value)
                            rel_dict["type"] = value.type
                            rel_dict["start_node"] = value.start_node.id
                            rel_dict["end_node"] = value.end_node.id
                            relationships.append(rel_dict)

                return {
                    "nodes": nodes,
                    "relationships": relationships,
                    "cypher": cypher_query,
                }

            except Exception as e:
                print(f"Error executing Cypher query: {e}")
                return {"nodes": [], "relationships": [], "error": str(e)}

    def _generate_cypher_query(self, query: str) -> str:
        """
        Generate a Cypher query from natural language using LLM.

        Args:
            query: User query in natural language

        Returns:
            Cypher query string
        """
        prompt = f"""You are an expert in converting natural language questions about EU environmental policy into Cypher queries for Neo4j.

The knowledge graph has the following schema:

Nodes:
- Policy (properties: name, description, year, source_document, source_chunk_id)
- Target (properties: name, value, unit, deadline, description, source_document, source_chunk_id)
- Legislation (properties: name, type, year, description, source_document, source_chunk_id)
- Country (properties: name, iso_code)
- RenewableSource (properties: name, type)
- Sector (properties: name, description)

Relationships:
- MANDATES (Policy → Target)
- IMPLEMENTS (Legislation → Policy)
- APPLIES_TO (Target → Country)
- PROMOTES (Policy → RenewableSource)
- IMPACTS (Policy → Sector)
- SET_BY (Target → Legislation)
- SUPPORTS (Country → RenewableSource)

Question: {query}

Generate a Cypher query that retrieves relevant nodes and relationships to answer this question.
Return ONLY the Cypher query, nothing else. The query should return nodes and relationships.
Use MATCH and RETURN statements. Limit results to 10 items.

Example format:
MATCH (p:Policy)-[r:MANDATES]->(t:Target)
WHERE p.name CONTAINS 'Green Deal'
RETURN p, r, t
LIMIT 10
"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Cypher query generation expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
            )

            cypher = chat_completion.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()

            return cypher

        except Exception as e:
            print(f"Error generating Cypher query: {e}")
            return ""

    def generate_explainable_answer(
        self, query: str, vector_results: List[Dict], graph_results: Dict
    ) -> Dict[str, any]:
        """
        Generate an explainable answer using both vector and graph context.

        Args:
            query: User query
            vector_results: Results from vector search
            graph_results: Results from graph search

        Returns:
            Dictionary with answer, explanation, and citations
        """
        # Prepare context from vector search
        vector_context = "\n\n".join(
            [
                f"[Source: {r['metadata']['source_document']}, Chunk: {r['metadata']['chunk_id']}]\n{r['text']}"
                for r in vector_results
            ]
        )

        # Prepare context from graph search
        graph_context = self._format_graph_context(graph_results)

        # Create prompt for explainable generation
        prompt = f"""You are an expert assistant on EU environmental policy and renewable energy.

Answer the following question using the provided context from both text documents and a knowledge graph.

Question: {query}

=== TEXT CONTEXT ===
{vector_context}

=== KNOWLEDGE GRAPH CONTEXT ===
{graph_context}

Instructions:
1. Provide a clear, accurate answer to the question
2. Explain your reasoning by referencing specific entities and relationships from the knowledge graph
3. Cite the specific text sources that support your answer
4. Structure your response as follows:

ANSWER:
[Your answer here]

EXPLANATION:
[Explain the reasoning path through the knowledge graph, mentioning specific nodes and relationships]

CITATIONS:
[List the source documents and chunk IDs that support your answer]
"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert explainable AI assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )

            response_text = chat_completion.choices[0].message.content

            # Parse the response
            parsed = self._parse_response(response_text)

            return {
                "answer": parsed["answer"],
                "explanation": parsed["explanation"],
                "citations": parsed["citations"],
                "graph_data": graph_results,
                "vector_sources": vector_results,
            }

        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": "Error generating answer",
                "explanation": str(e),
                "citations": [],
                "graph_data": {},
                "vector_sources": [],
            }

    def _format_graph_context(self, graph_results: Dict) -> str:
        """Format graph results into readable context."""
        if not graph_results.get("nodes"):
            return "No relevant graph data found."

        context = "Relevant entities and relationships:\n\n"

        # Format nodes
        context += "Entities:\n"
        for node in graph_results["nodes"][:10]:  # Limit to 10
            labels = ", ".join(node.get("labels", []))
            name = node.get("name", "Unknown")
            context += f"- {labels}: {name}\n"
            for key, value in node.items():
                if key not in [
                    "labels",
                    "id",
                    "name",
                    "source_document",
                    "source_chunk_id",
                ]:
                    context += f"  {key}: {value}\n"

        # Format relationships
        if graph_results.get("relationships"):
            context += "\nRelationships:\n"
            for rel in graph_results["relationships"][:10]:
                rel_type = rel.get("type", "UNKNOWN")
                context += f"- {rel_type}\n"

        return context

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """Parse the LLM response into structured components."""
        sections = {"answer": "", "explanation": "", "citations": ""}

        current_section = None
        lines = response_text.split("\n")

        for line in lines:
            line_upper = line.strip().upper()
            if line_upper.startswith("ANSWER:"):
                current_section = "answer"
                continue
            elif line_upper.startswith("EXPLANATION:"):
                current_section = "explanation"
                continue
            elif line_upper.startswith("CITATIONS:"):
                current_section = "citations"
                continue

            if current_section:
                sections[current_section] += line + "\n"

        return {k: v.strip() for k, v in sections.items()}

    def query(self, query: str, top_k: int = 3) -> Dict[str, any]:
        """
        Main query method combining all retrieval and generation steps.

        Args:
            query: User query
            top_k: Number of vector search results

        Returns:
            Complete explainable response
        """
        print(f"Processing query: {query}")

        # Step 1: Vector search
        print("Performing vector search...")
        vector_results = self.vector_search(query, top_k)

        # Step 2: Graph search
        print("Performing graph search...")
        graph_results = self.graph_search(query)

        # Step 3: Generate explainable answer
        print("Generating explainable answer...")
        response = self.generate_explainable_answer(
            query, vector_results, graph_results
        )

        return response


def main():
    """Main execution function for testing."""
    # Neo4j connection
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # Initialize RAG system
    rag = ExplainableRAGSystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Load chunks
    from data_preparation import main as prepare_data

    chunks = prepare_data()
    rag.load_and_index_chunks(chunks)

    # Test query
    test_query = (
        "What are the renewable energy targets set by the European Green Deal for 2030?"
    )
    response = rag.query(test_query)

    print("\n=== ANSWER ===")
    print(response["answer"])
    print("\n=== EXPLANATION ===")
    print(response["explanation"])
    print("\n=== CITATIONS ===")
    print(response["citations"])

    rag.close()


if __name__ == "__main__":
    main()
