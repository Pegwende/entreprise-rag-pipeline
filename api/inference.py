import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import psycopg2
import logging
import json

logger = logging.getLogger(__name__)

class RAGResponseSchema(BaseModel):
    answer: str = Field(description="The synthesized answer derived from the provided context.")
    source_documents: List[str] = Field(description="List of document IDs used to formulate the answer.")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0.")

class EnterpriseInferenceService:
    def __init__(self):
        # Initialize the official Google Gen AI client with SSL verification bypassed for both sync and async clients
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                client_args={"verify": False},
                async_client_args={"verify": False}
            )
        )
        self.model_name = "gemini-2.5-flash"
 
    def retrieve_relevant_chunks(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Executes a real vector similarity search over PostgreSQL using pgvector and Gemini embeddings.
        """
        logger.info(f"[RETRIEVAL] Querying pgvector database for query: '{query}'")

        # Generate query embedding using Gemini
        embed_response = self.client.models.embed_content(
            model="gemini-embedding-2",
            contents=query,
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )
        query_vector = embed_response.embeddings[0].values

        # Connect to PostgreSQL and query using cosine distance (<=>)
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "enterprise_rag"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )

        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT doc_id, department, chunk_id, chunk_text, 1 - (embedding <=> %s::vector) AS similarity
                    FROM document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                cursor.execute(sql, (query_vector, query_vector, top_k))
                rows = cursor.fetchall()

                # Map database rows to expected dictionary structures
                context_chunks = []
                for row in rows:
                    context_chunks.append({
                        "doc_id": row[0],
                        "department": row[1],
                        "chunk_id": row[2],
                        "chunk_text": row[3],
                        "similarity": float(row[4])
                    })
                return context_chunks
        finally:
            conn.close()

    def generate_grounded_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Orchestrates the Gemini API call using retrieved context and enforces a strict structural schema.
        """

        if not context_chunks:
            logger.info("[WARNING] Retrieval step returned no relevant chunks for the query.")
            return {
                "answer": "I am sorry, but I could not find any internal documents relevant to your query to safely answer this question.",
                "source_documents": [],
                "confidence_score": 0.0
            }
        try:
            logger.info("[INFERENCE] Formatting context and invoking Gemini inference...")

            # Format context string from retrieved chunks
            context_text = "\n---\n".join([f"Doc ID: {c['doc_id']} ({c['department']}): {c['chunk_text']}" for c in context_chunks])
            source_ids = list(set([c['doc_id'] for c in context_chunks]))

            prompt = f"""
            You are an enterprise AI assistant for Workday. Answer the user's question accurately
            based ONLY on the provided context below. Do not assume or extrapolate.

            Context:
            {context_text}

            User Question: {query}
            """

            # Invoke Gemini with Structured Outputs ensuring it adheres to RAGResponseSchema
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RAGResponseSchema,
                    temperature=0.1

                ),
            )

            # Parse the structured JSON response back into a standard dictionary
            result_data = json.loads(response.text)
            return result_data
            
        except Exception as e:
            logger.info(f"[ERROR] Inference execution failed: {str(e)}")
            return {
                "answer": "An unexpected error occurred while processing your request. Please try again later.",
                "source_documents": [],
                "confidence_score": 0.0
                }