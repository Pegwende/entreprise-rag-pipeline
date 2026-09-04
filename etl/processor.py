import json
import pandas as pd
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class EnterpriseETLPipeline:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.DataFrame()

    def extract(self) -> pd.DataFrame:
        """Extract raw data from JSON source."""
        print("[EXTACT] Loading raw enterprise documents ...")
        with open(self.file_path, "r" , encoding='utf-8') as f:
            data = json.load(f)
        self.df = pd.DataFrame(data)
        return self.df

    def transform_and_clean(self) -> pd.DataFrame:
        """
        Clean missing values, normalize text formatting,
        and structure metadata using Pandas.
        """
        if self.df.empty:
            raise ValueError("DataFrame is empty. Run extract() first. ")

        print("[TRANSFORM] Cleaning data and structuring metadata...")

        #Handle missing timestamps by filling with a default or current date
        self.df['timestamp'] = self.df['timestamp'].fillna("2026-01-01T00:00:00Z")

        #Handle missing departments if any
        self.df['department'] = self.df['department'].fillna("General")

        #Normalize text formatting: strip whitespaces and collapse double spaces
        self.df['content'] = (
            self.df['content']
            .str.strip()
            .replace(r'\s+', ' ', regex=True)
        )

        #Add metadata fields required for enterprise tracking
        self.df['processed_at'] = pd.Timestamp.utcnow().isoformat()
        self.df['char_length'] = self.df['content'].str.len()
        
        return self.df

    def chunk_documents(self, chunk_size: int = 150) -> List[Dict[str, Any]]:
        """
        Split text blocks into clean chunks for vector embedding and retrieval.
        """
        print(f"[TRANSFORM] Chunking documents (max size: {chunk_size} chars)...")
        chunked_records = []

        for _, row in self.df.iterrows():
            text = row['content']
            doc_id = row['doc_id']
            department = row['department']

            #Simple character-based sliding window chunking
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size]
                chunked_records.append({
                    "doc_id": doc_id,
                    "department": department,
                    "chunk_id": f"{doc_id}_chunk_{i // chunk_size}",
                    "chunk_text": chunk_text
                })
        return chunked_records

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simulates or connects to an embedding model to convert text chunks into vectors.
        In production, this calls OpenAI/Gemini embedding endpoints.
        """
        print("[ML EMBEDDING] Generating vector embeddings for chunks...")
        
        # For a robust offline fallback or initial test, we can define the structure 
        # that integrates with OpenAI/Gemini API clients.
        embedded_records = []
        for chunk in chunks:
            # Simulated 4-dimensional vector for local testing; 
            # in production, call client.embeddings.create(input=chunk['chunk_text'], model="text-embedding-3-small")
            simulated_vector = [0.123, -0.456, 0.789, 0.321] 
            
            embedded_records.append({
                **chunk,
                "embedding": simulated_vector
            })
            
        print(f"[ML EMBEDDING] Generated embeddings for {len(embedded_records)} chunks.")
        return embedded_records 

if __name__ == "__main__":
    pipeline = EnterpriseETLPipeline("data/mock_enterprise_docs.json")
    data = pipeline.extract()
    print(pipeline.df)
    print("--------------------------")
    cleaned_df = pipeline.transform_and_clean()
    print(pipeline.df)
    print("--------------------------")
    chunks = pipeline.chunk_documents(chunk_size=120)
    print(f"\nSuccessfully processed and created {len(chunks)} text chunks ready for embedding.")
    print(chunks[0])
    print(chunks[1])
    print(chunks[2])
    print(chunks[3])
    print("--------------------------")
    embedded_chunks = pipeline.generate_embeddings(chunks)
    print("\nFirst embedded record with vector:")
    print(embedded_chunks[0])
    print(embedded_chunks[1])
    print(embedded_chunks[2])
    print(embedded_chunks[3])
    print(embedded_chunks[4])
    print(embedded_chunks[5])


