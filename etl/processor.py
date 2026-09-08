import os
import logging
import psycopg2
from google import genai
from dotenv import load_dotenv
import json
import pandas as pd
from typing import List, Dict, Any
from pgvector.psycopg2 import register_vector
import certifi
import httpx
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg2.extras import execute_values
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

# Configure logging so info messages print to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# Initialize the official Google Gen AI client with SSL verification bypassed for local dev
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        client_args={"verify": False}
    )
)

class EnterpriseETLPipeline:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.DataFrame()

    def extract(self) -> pd.DataFrame:
        """Extract raw data from JSON source."""
        logger.info("[EXTRACT] Loading raw enterprise documents ...")
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

        logger.info("[TRANSFORM] Cleaning data and structuring metadata...")

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
        self.df['processed_at'] = pd.Timestamp.now('UTC').isoformat()
        self.df['char_length'] = self.df['content'].str.len()
        
        return self.df

    def chunk_documents(self, chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split text blocks into a recursive character splitter with overlap
        to preserve semantic boundaries for vector embeddings.
        """
        logger.info(f"[TRANSFORM] Chunking documents (size: {chunk_size}, overlap: {chunk_overlap})...")
        chunked_records = []

        # Initialize a production-grade splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators = ["\n\n", "\n", " ", ""]
        )

        for _, row in self.df.iterrows():
            text = row['content']
            doc_id = row['doc_id']
            department = row['department']

            #Generate intelligent chunks
            docs = splitter.create_documents(
                texts=[text],
                metadatas=[{"doc_id": doc_id, "department": department}]
            )

            for idx, doc in enumerate(docs):
                chunked_records.append({
                    "doc_id": doc_id,
                    "department": department,
                    "chunk_id": f"{doc_id}_chunk_{idx}",
                    "chunk_text": doc.page_content
                })
        return chunked_records

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calls the live Gemini API to generate 768-dimensional vector embeffings."""
        logger.info("[ML EMBEDDING] Generating live Gemini vector embeddings for chunks...")
        embedded_records = []

        for chunk in chunks:
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=chunk['chunk_text'],
                    config=types.EmbedContentConfig(
                        output_dimensionality=768
                    )
                )
                vector = response.embeddings[0].values

                embedded_records.append({
                    **chunk,
                    "embedding": vector
                })
            except Exception as e:
                logger.error(f"Failed to generate embedding for {chunk['chunk_id']}: {e}")
                raise
            
        logger.info(f"[ML EMBEDDING] Generated live embeddings for {len(embedded_records)} chunks.")
        return embedded_records 

    def create_table_if_not_exists(self):
        """Creates the document_chunks table and enables pgvector if missing."""
        logger.info("[INIT] Ensuring PostgreSQL table and vector extension exist...")
        db_url = os.getenv("DATABASE_URL", f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}")
        
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        try:
            # 1. Create the extension first so PostgreSQL knows the 'vector' type
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

            # 2. Register vector type for psycopg2 mapping
            register_vector(conn)

            # 3. Create table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255),
                    department VARCHAR(255),
                    chunk_id VARCHAR(255) UNIQUE,
                    chunk_text TEXT,
                    embedding vector(768)
                );
            """)
            conn.commit()
            logger.info("[INIT] Database table ready.")
        except Exception as e:
            conn.rollback()
            logger.error(f"[INIT] Failed to create table: {e}")
            raise
        finally:
            cur.close()
            conn.close()

    def load_to_postgres(self, embedded_records: List[Dict[str, Any]]):
        """Persists chunks and vectors efficiently using batch execution. """ 
        logger.info("[LOAD] Storing embedded records into PostgreSQL in bulk...")

        db_url = os.getenv("DATABASE_URL", f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}")

        conn = psycopg2.connect(db_url)
        register_vector(conn)
        cur = conn.cursor()

        try:
            # Prepare tuples for bulk insertion
            values = [
                (
                    record['doc_id'],
                    record['department'],
                    record['chunk_id'],
                    record['chunk_text'],
                    record['embedding']
                )
                for record in embedded_records
            ]

            # Use execute_values for high-perfomance batch loading
            query =  """
                INSERT INTO document_chunks (doc_id, department, chunk_id, chunk_text, embedding) 
                VALUES %s
                ON CONFLICT (chunk_id) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding = EXCLUDED.embedding;
            """

            execute_values(cur, query, values)
            conn.commit()
            logger.info(f"[LOAD] Successfully batch-saved {len(embedded_records)} records to PostgreSQL.")
        except Exception as e:
            conn.rollback()
            logger.error(f"[LOAD] Database insertion failed: {e}")
            raise
        finally:
            cur.close()
            conn.close()



if __name__ == "__main__":
    pipeline = EnterpriseETLPipeline("data/mock_enterprise_docs.json")
    data = pipeline.extract()
    cleaned_df = pipeline.transform_and_clean()
    chunks = pipeline.chunk_documents(chunk_size=300, chunk_overlap=50)
    logger.info(f"\nSuccessfully processed and created {len(chunks)} text chunks ready for embedding.")
    
    # Ensure database table and vector extension exist before loading
    pipeline.create_table_if_not_exists()
    
    embedded_chunks = pipeline.generate_embeddings(chunks)

    # Push live embeddings directly into PostgreSQL
    pipeline.load_to_postgres(embedded_chunks)
    logger.info("\nFirst embedded record with vector:")
    logger.info(embedded_chunks[0])



