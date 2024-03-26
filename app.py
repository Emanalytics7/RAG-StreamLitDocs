from document_processor import DocumentProcessor
from openai_ import OpenAIModel
from vector_store_manager import ChromaDBStorage
import os

# Configuration and Initialization
openai_api_key = "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"
pdf_file_path = "sample-pdf-file.pdf"

# Initialize components
doc_processor = DocumentProcessor()
openai_model = OpenAIModel(api_key=openai_api_key)
vector_store = ChromaDBStorage(db_path='./chroma_db', collection_name='systems')

# Step 1: Extract text from PDF
text = doc_processor.get_text(open(pdf_file_path, 'rb'))

# Step 2: Generate an embedding for the extracted text
embedding = openai_model.create_embedding(text)

# Step 3: Store the embedding in ChromaDB
# Assuming the store_embedding method takes a document ID and its embedding
doc_id = os.path.basename(pdf_file_path)  # Use the file name as a simple document ID
vector_store.store_embedding(doc_id, embedding)

# Step 4: Query ChromaDB for similar embeddings
# For simplicity, we use the same embedding as the query; in practice, you might query with different text
similar_embeddings = vector_store.query_embedding(embedding, n_results=5)

# Output the result of the query
# This part depends on how similar_embeddings are structured and what information they contain
print("Similar Embeddings Found:", similar_embeddings)
