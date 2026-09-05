from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from api.inference import EnterpriseInferenceService, RAGResponseSchema
from etl.processor import EnterpriseETLPipeline

app = FastAPI(
    title="Workday Enterprise RAG & Inference Pipeline",
    version="1.0.0",
    description="Production-grade AI inference and retrieval service for enterprise document processing."
)

#Initialize services and load processed data on startup
pipeline = EnterpriseETLPipeline("data/mock_enterprise_docs.json")
pipeline.extract()
pipeline.transform_and_clean()
CHUNKS = pipeline.chunk_documents(chunk_size=120)

inference_service = EnterpriseInferenceService()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "loaded_chunks_count": len(CHUNKS)}

@app.post("/api/v1/query", response_model=RAGResponseSchema)
def query_enterprise_kb(request: QueryRequest):
    try:
        # Step 1: Information Retrieval
        relevant_chunks = inference_service.retrieve_relevant_chunks(request.query, CHUNKS, top_k=2)

        if not relevant_chunks:
            raise HTTPException(status_code=404, detail="No relevant enterprise documentation found.")

        # Step 2: LLM Inference & Generation
        result = inference_service.generate_grounded_answer(request.query, relevant_chunks)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

