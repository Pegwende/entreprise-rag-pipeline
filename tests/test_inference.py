import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etl.processor import EnterpriseETLPipeline
from api.inference import EnterpriseInferenceService

# 1. Run the full ETL pipeline cleanly in sequence
etl = EnterpriseETLPipeline("data/mock_enterprise_docs.json")
etl.extract()
etl.transform_and_clean()
chunks = etl.chunk_documents(chunk_size=120)

# 2. Pass the resulting chunks directly into the inference service
inference = EnterpriseInferenceService()
query = "What are the remote work rules?"

retrieved_chunks = inference.retrieve_relevant_chunks(query, chunks, top_k=2)
response = inference.generate_grounded_answer(query, retrieved_chunks)

print("\n--- End-to-End Pipeline Output ---")
print(response)