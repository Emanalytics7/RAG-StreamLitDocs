import streamlit as st
from doc_processor import DocumentProcessor


# setup configuration
st.set_page_config(page_title='DocBot', layout='wide', page_icon='icons8-chatbot-64.png')
def main():
    """
    This function sets up the Streamlit application 
    and handles the user interactions.

    """
    st.title('DOCBOT 👾✨')

    with st.sidebar:
        st.title('Hi there! 💁‍♀️')
        st.markdown('Drop your docs here and watch us do magic!😉')
        st.image('icons8-export-pdf-96.png', width=110)

        # file upload
        uploaded_file = st.file_uploader('Choose a document file (.txt, .pdf, .docx)', type=['txt', 'pdf', 'docx'])
        openai_api_key = st.text_input('Enter your OpenAI API Key:', type='password')
        # validate API key format
        if openai_api_key and not openai_api_key.startswith('sk'):
            st.error('Whoopsie! Wrong API, wrong dimension!')

    st.markdown('### Ask me anything about your document!')
    query_text = st.text_input('Your question:', placeholder='What would you like to know?')

    if openai_api_key.startswith('sk') and uploaded_file and query_text:
        process_button = st.button('Ask', key='process_button')

        if process_button:
            with st.spinner('Thinking...'):
                doc_processor = DocumentProcessor(openai_api_key)
                try:
                    response = doc_processor.generate_response(uploaded_file, query_text)
                    st.success('Response:')
                    st.write(response)
                except Exception as e:
                    st.error(f'An error occurred: {str(e)}')

    elif not uploaded_file and not query_text:
        st.markdown('You didn\'t upload a document. I\'m pretty good, but I\'m not psychic... yet!')
        if not query_text and st.button('Ask', key='no_query_text'):
           st.markdown('You\'re testing my mind-reading skills, aren\'t you? Go on, type a question!🤔 ')

if __name__ == "__main__":
    main()
