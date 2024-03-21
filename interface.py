import streamlit as st

def initialize_interface():
    st.set_page_config(page_title="Ask your PDF", layout="wide")
    st.header("Ask your PDF 🤔")

def user_file_uploader():
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your files", type=['pdf', 'docx'], accept_multiple_files=True)
        return uploaded_files

def user_api_key_input():
    openai_api_key = st.sidebar.text_input('Enter your OpenAI API Key:', type='password')
    return openai_api_key

def process_button():
    return st.sidebar.button('Process Files')

def user_query_input():
    return st.text_input('Ask a question related to your uploaded files:', '')

def display_response(response):
    st.markdown("### Answer:")
    st.info(response)
