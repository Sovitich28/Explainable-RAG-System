"""
Data Preparation Module
Handles document loading, cleaning, and chunking for the RAG system.
"""

import os
import re
from pathlib import Path
from typing import Dict, List


class DocumentProcessor:
    """Process raw documents into clean, chunked text."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: str) -> str:
        """
        Load a document from file.

        Args:
            file_path: Path to the document

        Returns:
            Document content as string
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep punctuation
        text = re.sub(r"[^\w\s.,;:!?()\-\'\"€%]", "", text)
        # Normalize line breaks
        text = text.replace("\n\n\n", "\n\n")
        return text.strip()

    def chunk_by_paragraph(self, text: str) -> List[Dict[str, any]]:
        """
        Chunk text by paragraphs with semantic coherence.

        Args:
            text: Cleaned text content

        Returns:
            List of chunk dictionaries with metadata
        """
        # Split by double newlines (paragraphs) or headers
        sections = re.split(r"\n#{1,3}\s+", text)

        chunks = []
        chunk_id = 0

        for section in sections:
            # Extract header if present
            lines = section.split("\n")
            header = lines[0].strip() if lines else ""
            content = "\n".join(lines[1:]) if len(lines) > 1 else section

            # Split long sections into smaller chunks
            if len(content) > self.chunk_size:
                words = content.split()
                current_chunk = []
                current_length = 0

                for word in words:
                    current_chunk.append(word)
                    current_length += len(word) + 1

                    if current_length >= self.chunk_size:
                        chunk_text = " ".join(current_chunk)
                        chunks.append(
                            {
                                "chunk_id": f"chunk_{chunk_id}",
                                "text": chunk_text,
                                "header": header,
                                "length": len(chunk_text),
                            }
                        )
                        chunk_id += 1

                        # Keep overlap
                        overlap_words = int(
                            len(current_chunk) * (self.chunk_overlap / self.chunk_size)
                        )
                        current_chunk = (
                            current_chunk[-overlap_words:] if overlap_words > 0 else []
                        )
                        current_length = sum(len(w) + 1 for w in current_chunk)

                # Add remaining words
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        {
                            "chunk_id": f"chunk_{chunk_id}",
                            "text": chunk_text,
                            "header": header,
                            "length": len(chunk_text),
                        }
                    )
                    chunk_id += 1
            else:
                # Small section, keep as single chunk
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id}",
                        "text": content.strip(),
                        "header": header,
                        "length": len(content),
                    }
                )
                chunk_id += 1

        return chunks

    def process_document(self, file_path: str, output_dir: str) -> List[Dict[str, any]]:
        """
        Process a document from loading to chunking.

        Args:
            file_path: Path to input document
            output_dir: Directory to save processed chunks

        Returns:
            List of processed chunks with metadata
        """
        # Load and clean
        raw_text = self.load_document(file_path)
        clean_text = self.clean_text(raw_text)

        # Chunk
        chunks = self.chunk_by_paragraph(clean_text)

        # Add source metadata
        doc_name = Path(file_path).stem
        for chunk in chunks:
            chunk["source_document"] = doc_name

        # Save chunks
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{doc_name}_chunks.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(f"=== {chunk['chunk_id']} ===\n")
                f.write(f"Header: {chunk['header']}\n")
                f.write(f"Source: {chunk['source_document']}\n")
                f.write(f"{chunk['text']}\n\n")

        print(f"Processed {len(chunks)} chunks from {doc_name}")
        return chunks


def main():
    """Main execution function."""
    # Initialize processor
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

    # Process all documents in raw data directory
    # Use absolute paths relative to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    chunks_dir = os.path.join(base_dir, "data", "chunks")

    # Ensure directories exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)

    all_chunks = []
    if os.path.exists(raw_dir):
        for file_name in os.listdir(raw_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(raw_dir, file_name)
                chunks = processor.process_document(file_path, chunks_dir)
                all_chunks.extend(chunks)
    else:
        print(f"Warning: Raw data directory not found at {raw_dir}")

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    main()
