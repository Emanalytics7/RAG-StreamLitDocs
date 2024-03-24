import streamlit as st
from document_processor import DocumentProcessor  
from text_chunker import TextChunker
from vector_store_manager import VectorStoreManager
from conversation_handler import ConversationChainManager  
def main():
    st.title('Document Query Application')
    uploaded_files = st.file_uploader("Upload Documents", type=['pdf', 'docx'], accept_multiple_files=True)
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

    if uploaded_files and openai_api_key:
        process_button = st.button('Process Documents')
        
        if process_button:
            vector_store_manager = VectorStoreManager(collection_name="pdf", openai_api_key=openai_api_key)
            conversation_chain_manager = ConversationChainManager(llm="gpt-3.5-turbo", retriever=vector_store_manager, openai_api_key=openai_api_key)
            
            for uploaded_file in uploaded_files:
                text = DocumentProcessor.get_text(uploaded_file)
                text_chunker = TextChunker()
                chunks = text_chunker.chunk_text(text)
                
                vector_store_manager.store_embeddings(chunks)
            
            st.success("Documents processed and embeddings stored.")
    
    user_query = st.text_input("Enter your query:")
    if user_query:
        answer = conversation_chain_manager.query_gpt3_5_turbo(user_query)
        st.write("Answer:", answer)

if __name__ == "__main__":
    main()
