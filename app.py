import openai
import streamlit as st
from document_processor import DocumentProcessor
from text_chunker import TextChunker
from vector_store_manager import VectorStoreManager
from conversation_handler import ConversationChainManager
import interface

def main():
    interface.initialize_interface()
    uploaded_files = interface.user_file_uploader()
    openai_api_key = interface.user_api_key_input()
    
    if interface.process_button() and uploaded_files:
        for uploaded_file in uploaded_files:
            text = DocumentProcessor.get_text(uploaded_file)
            text_chunker = TextChunker()
            chunks = text_chunker.chunk_text(text)
            
            vector_store_manager = VectorStoreManager(collection_name="YourCollectionName")
            vector_store_manager.store_embeddings(chunks, embeddings_model)
            
        st.success("Files processed successfully.")

    query = interface.user_query_input()
    if query:
        conversation_chain_manager = ConversationChainManager(llm=model, retriever=vector_store_manager)
        conversation_chain = conversation_chain_manager.get_conversation_chain()
        response = conversation_chain.handle_query(query)
        interface.display_response(response)

if __name__ == "__main__":
    main()
