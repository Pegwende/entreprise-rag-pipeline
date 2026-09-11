Enterprise Workday RAG Pipeline
A production-grade Retrieval-Augmented Generation (RAG) backend built with FastAPI, PostgreSQL 18 (pgvector), and the Google Gen AI SDK (gemini-3.6-flash & gemini-embedding-2).

Architecture
Ingestion & ETL: Normalizes enterprise documents, applies sliding-window chunking, and batch-generates 768-dimensional embeddings via Gemini.

Retrieval: Executes lightning-fast semantic similarity searches in PostgreSQL using pgvector and cosine distance.

Inference: Combines retrieved context with strict Pydantic schemas (gemini-3.6-flash) to guarantee reliable, grounded JSON API contracts.

Tech Stack
Backend: FastAPI, Uvicorn, Pydantic v2, Psycopg2

Database: PostgreSQL 18 with pgvector

AI / LLM: Google Gen AI SDK (gemini-3.6-flash, gemini-embedding-2)

Quickstart (Docker Compose)
Create a .env file in the root directory:
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=enterprise_rag
GEMINI_API_KEY=your_gemini_api_key_here

Build and launch with Docker:
docker-compose up --build

API Endpoints
GET /health: Verifies database connection and active chunk counts.

POST /api/v1/query: Executes semantic retrieval and returns a structured RAG answer.