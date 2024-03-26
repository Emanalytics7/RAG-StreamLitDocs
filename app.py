import openai
from document_processor import DocumentProcessor
from text_chunker import TextChunker
from vector_store_manager import ChromaDBStorage
import numpy as np

# Assuming you have set these up
openai.api_key = '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'

def generate_embedding(text):
    """Generate an embedding for a piece of text using OpenAI's API."""
    response = openai.embeddings.create(
        input=[text],
        engine="text-embedding-3-large"
    )
    return response.data[0].embedding

def process_document_and_store_chunks(file_path):
    """Extract text, chunk it, generate embeddings, and store them."""
    text = DocumentProcessor.get_text(file_path)
    chunks = TextChunker().chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        ChromaDBStorage().store_embedding(str(i), np.array(embedding))

def query_and_generate_response(query):
    """Find relevant document chunks and use them to generate a response."""
    query_embedding = generate_embedding(query)
    results = ChromaDBStorage().query_embedding(np.array(query_embedding))
    
    # Assuming `results` contains the IDs of the relevant chunks
    relevant_chunks = [ChromaDBStorage().retrieve_chunk_text(chunk_id) for chunk_id in results]  # Placeholder method
    combined_context = " ".join(relevant_chunks)
    
    response = openai.chat.completions.create(
        engine="gpt-3.5-turbo",
        prompt=f"Based on the following information: {combined_context}\n\nGenerate a summary:",
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].text

# Example usage
if __name__ == "__main__":
    file_path = 'sample-pdf-file.pdf'  # or .docx
    process_document_and_store_chunks(file_path)

    query = "What is the summary of the document?"
    response = query_and_generate_response(query)
    print(response)
