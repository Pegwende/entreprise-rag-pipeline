import pytest
from fastapi.testclient import TestClient
from api.main import app
from etl.processor import EnterpriseETLPipeline

client = TestClient(app)

def test_etl_pipeline_cleaning():
    """Verify ETL properly cleans missing fields and formats text."""
    pipeline = EnterpriseETLPipeline("data/mock_enterprise_docs.json")
    df = pipeline.extract()
    cleaned_df = pipeline.transform_and_clean()

    #Assert missing timestamp was handled
    assert not cleaned_df['timestamp'].isnull().any()
    #Assert string whitespace is stripped
    assert not cleaned_df['content'].str.startswith(' ').any()

def test_health_endpoint():
    """Verify API health check returns active status and chunk counts."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["loaded_chunks_count"] > 0

def test_query_endpoint_valid():
    """Verify query endpoint returns schema-compliant response."""
    payload = {"query": "expense reports travel"}
    response = client.post("/api/v1/query", json=payload)

    assert response.status_code == 200
    data = response.json()

    #Check schema keys required by Pydantic response model
    assert "answer" in data
    assert "source_documents" in data
    assert "confidence_score" in data
    assert isinstance(data["confidence_score"], float)
