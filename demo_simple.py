"""
Simple Demo Script
Demonstrates the RAG system without requiring Neo4j.
Uses only vector search and LLM generation (via Groq).
"""

import os
import sys
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()


class SimpleRAGDemo:
    """Simplified RAG demonstration without knowledge graph."""

    def __init__(self):
        """Initialize the demo system."""
        # Configure Groq
        api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
                print("✅ Groq client initialized.")
            except Exception as e:
                print(f"⚠️ Failed to initialize Groq: {e}")
        else:
            print("⚠️ GROQ_API_KEY not found. AI generation will be skipped.")

        print("Loading embedding model (this runs locally)...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = []
        self.chunk_embeddings = None

    def load_chunks(self, chunks_file: str):
        """Load processed chunks from file."""
        print(f"Loading chunks from {chunks_file}...")

        if not os.path.exists(chunks_file):
            print(f"❌ Error: File not found: {chunks_file}")
            return

        with open(chunks_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse chunks
        chunk_sections = content.split("=== chunk_")

        for section in chunk_sections[1:]:  # Skip first empty section
            lines = section.split("\n")
            chunk_id = "chunk_" + lines[0].split(" ===")[0]

            # Extract metadata and text
            header = ""
            source = ""
            text = []

            for line in lines[1:]:
                if line.startswith("Header:"):
                    header = line.replace("Header:", "").strip()
                elif line.startswith("Source:"):
                    source = line.replace("Source:", "").strip()
                elif line.strip() and not line.startswith("==="):
                    text.append(line)

            chunk_text = " ".join(text).strip()

            if chunk_text:
                self.chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "header": header,
                        "source_document": source,
                        "text": chunk_text,
                    }
                )

        print(f"✅ Loaded {len(self.chunks)} chunks")

        # Create embeddings
        print("Creating embeddings...")
        texts = [chunk["text"] for chunk in self.chunks]
        self.chunk_embeddings = self.embed_model.encode(texts, show_progress_bar=True)
        print("✅ Embeddings created!")

    def vector_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Perform vector similarity search."""
        # Encode query
        query_embedding = self.embed_model.encode([query])[0]

        # Calculate cosine similarity
        similarities = np.dot(self.chunk_embeddings, query_embedding) / (
            np.linalg.norm(self.chunk_embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                {"chunk": self.chunks[idx], "score": float(similarities[idx])}
            )

        return results

    def generate_answer(self, query: str, search_results: List[Dict]) -> Dict:
        """Generate an answer using LLM."""
        # Prepare context
        context = "\n\n".join(
            [
                f"[Source: {r['chunk']['source_document']}, Chunk: {r['chunk']['chunk_id']}, Score: {r['score']:.3f}]\n{r['chunk']['text']}"
                for r in search_results
            ]
        )

        # Create prompt
        prompt = f"""You are an expert assistant on EU environmental policy and renewable energy.

Answer the following question using the provided context from official EU documents.

Question: {query}

Context:
{context}

Instructions:
1. Provide a clear, accurate answer to the question
2. Explain your reasoning based on the information in the context
3. Cite the specific sources that support your answer
4. Structure your response as follows:

ANSWER:
[Your answer here]

EXPLANATION:
[Explain your reasoning, referencing specific information from the context]

CITATIONS:
[List the source documents and chunk IDs that support your answer]
"""

        if not self.client:
            return {
                "answer": "AI Generation Skipped (No API Key)",
                "explanation": "Please provide a GROQ_API_KEY in .env to generate an answer.",
                "citations": "N/A",
            }

        print("\nGenerating answer with Groq (Llama 3)...")
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert explainable AI assistant for EU environmental policy.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )

            response_text = chat_completion.choices[0].message.content

            # Parse response (simple parsing)
            return {
                "answer": response_text,
                "explanation": "See above.",
                "citations": "See above.",
            }

        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "answer": "Error generating answer.",
                "explanation": str(e),
                "citations": "N/A",
            }

    def query(self, question: str, top_k: int = 3):
        """End-to-end query pipeline."""
        print(f"\n{'='*60}\nQUESTION: {question}\n{'='*60}\n")

        # 1. Vector Search
        print("Performing vector search...")
        search_results = self.vector_search(question, top_k)

        print(f"Found {len(search_results)} relevant chunks:")
        for i, res in enumerate(search_results, 1):
            print(f"  {i}. {res['chunk']['chunk_id']} (score: {res['score']:.3f})")

        # 2. Generate Answer
        response = self.generate_answer(question, search_results)

        print(f"\n{'='*60}\nGENERATED ANSWER\n{'='*60}")
        if isinstance(response, dict):
            # If we returned a dict (our manual parsing or fallback)
            if "answer" in response and response["answer"] != response.get(
                "explanation"
            ):
                print(f"\nANSWER:\n{response['answer']}")
                print(f"\nEXPLANATION:\n{response['explanation']}")
                print(f"\nCITATIONS:\n{response['citations']}")
            else:
                # If the model returned the formatted text directly
                print(response["answer"])
        else:
            print(response)


def main():
    print("============================================================")
    print("SIMPLE RAG DEMO - EU Green Deal & Renewable Energy")
    print("============================================================")

    demo = SimpleRAGDemo()

    # Load data
    chunks_file = os.path.join("data", "chunks", "eu_green_deal_sample_chunks.txt")
    if not os.path.exists(chunks_file):
        print(f"Chunks file not found at {chunks_file}")
        print("Please run 'python src/data_preparation.py' first.")
        return

    demo.load_chunks(chunks_file)

    # Example query
    query = "What are the EU's renewable energy targets for 2030?"
    demo.query(query, top_k=3)


if __name__ == "__main__":
    main()
