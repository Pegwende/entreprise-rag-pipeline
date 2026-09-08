import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from api.inference import EnterpriseInferenceService, RAGResponseSchema
from etl.processor import EnterpriseETLPipeline
import psycopg2

app = FastAPI(
    title="Workday Enterprise RAG & Inference Pipeline",
    version="1.0.0",
    description="Production-grade AI inference and retrieval service for enterprise document processing."
)

#Initialize services and load processed data on startup
@app.on_event("startup")
def startup_event():
    try:
        pipeline = EnterpriseETLPipeline("data/mock_enterprise_docs.json")
        pipeline.extract()
        pipeline.transform_and_clean()
        chunks = pipeline.chunk_documents(chunk_size=300, chunk_overlap=50)
        embedded_chunks = pipeline.generate_embeddings(chunks)
        pipeline.create_table_if_not_exists()
        pipeline.load_to_postgres(embedded_chunks)
    except Exception as e:
        print(f"Startup ETL notice: {e}")

inference_service = EnterpriseInferenceService()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    """Query database chunk count dynamically using environment configuration."""
    try:
        db_url = os.getenv(
            "DATABASE_URL",
            f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'postgres')}@{os.getenv('POSTGRES_HOST', 'postgres')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'enterprise_rag')}"
        )
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM document_chunks;")
            count = cursor.fetchone()[0]
        conn.close()
        return {"status": "healthy", "database_chunks_count": count}
    except Exception as e:
        logger.error(f"Health check DB connection failed: {e}")
        return {"status": "degraded", "database_chunks_count": 0}

@app.post("/api/v1/query", response_model=RAGResponseSchema)
def query_enterprise_kb(request: QueryRequest):
    try:
        # Step 1: Information Retrieval via pgvector
        relevant_chunks = inference_service.retrieve_relevant_chunks(request.query, top_k=2)

        if not relevant_chunks:
            raise HTTPException(status_code=404, detail="No relevant enterprise documentation found.")

        # Step 2: LLM Inference & Generation via Gemini Structured Outputs
        result = inference_service.generate_grounded_answer(request.query, relevant_chunks)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

