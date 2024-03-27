import streamlit as st
from doc_processor import DocumentProcessor 

def main():
    st.set_page_config(page_title='Document QA Processor', layout='wide')

    # Header section
    st.title('📄✨ Document-Based Question Answering')
    st.markdown('Upload your document and ask any question related to its content.')

    # Sidebar for API Key Input
    with st.sidebar:
        st.header('Configuration')
        openai_api_key = st.text_input('Enter your OpenAI API Key:', type='password')

    # Main content area
    with st.container():
        uploaded_file = st.file_uploader('Choose a document file (.txt, .pdf, .docx)', type=['txt', 'pdf', 'docx'])
        query_text = st.text_input('Enter your question:', placeholder='What would you like to know?')

        if uploaded_file and query_text and openai_api_key:
            process_button = st.button('Process Document')

            if process_button:
                with st.spinner('Processing...'):
                    doc_processor = DocumentProcessor(openai_api_key)
                    try:
                        response = doc_processor.generate_response(uploaded_file, query_text)
                        st.success('Response:')
                        st.write(response)
                    except Exception as e:
                        st.error(f'An error occurred: {str(e)}')
        else:
            st.warning('Please upload a document and enter a question to proceed.')

if __name__ == "__main__":
    main()
