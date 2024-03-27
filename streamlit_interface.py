import streamlit as st
from document_processor import DocumentProcessor
from text_chunker import TextChunker
from text_embedder import TextEmbedder
import os
from vector_store_manager import ChromaDBStorage
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

class QueryHandler:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, context, question, model="gpt-3.5-turbo"):
        """
        Generates a response to a question based on the provided context.
        
        Args:
        - context: The text context to inform the response.
        - question: The user's question.
        - model: The model to use for generating the response.
        
        Returns:
        - The generated response as a string.
        """
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response = self.client.Completions.create(
            model=model,
            prompt=prompt,
            temperature=0.5,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"]
        )
        return response.choices[0].text.strip()


db_storage = ChromaDBStorage()
text_embedder = TextEmbedder(api_key=os.getenv['OPENAI_API_KEY'])
st.title('PDF Document Query System')

uploaded_file = st.file_uploader('Upload your Document!', type=['pdf', 'docx'])
if uploaded_file is not None:
    text = DocumentProcessor.get_text(uploaded_file)
    chunker = TextChunker()
    chunks = chunker.chunk_text(text)

    for i, chunk in enumerate(chunks, start=1):
        chunk_id = f'{uploaded_file.name}_chunk_{i}'
        embedding = text_embedder.generate_embedding(chunk, model='text-embedding-3-small')
        db_storage.store_embedding(embedding, chunk, {'source': 'uploaded', 'chunk_index': i}, chunk_id)
    st.success('Document uploaded successfully!')


query_handler = QueryHandler(api_key=os.getenv('OPENAI_API_KEY'))

user_query = st.text_input('Enter your query:')
if user_query:
    query_embedding = text_embedder.generate_embedding(user_query, model='text-embedding-3-small')
    similar_doc_ids = db_storage.query_embedding(query_embedding, n_results=3)
    if similar_doc_ids:
        combined_context=''
        for doc_id in similar_doc_ids:
            document, metadata = db_storage.retrieve_text(doc_id)
            combined_context += f'{document}\n\n'
            response = query_handler.generate_response(context=combined_context, question=user_query)
        
        st.write("Generated Response:", response)
    else:
        st.write("No relevant document chunks found.")

