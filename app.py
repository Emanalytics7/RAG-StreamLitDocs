import os
import io
import pandas as pd
from typing import List
from docx import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 🤔")

    if 'conversation' not in st.session_state:
        st.session_state.conversion = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    if 'processComplete' not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader('Upload your files', type=['pdf', 'doc'], accept_multiple_files=True)
        openai_api_key = st.text_input('OpenAI API Key', key='chatbot_api_key', type='password')
        process = st.button('Process')
    if process:
        if not openai_api_key:
            st.info('Please add your OpenAI api key to proceed further')
            st.stop()

        files_text = get_files_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.converstation = get_conversation_chain(vectorstore, openai_api_key)

        st.session_state.processComplete = True

        if st.session_state.processComplete == True:
            user_question = st.chat_input('Ask Question related to your uploaded file')
            if user_question:
                 handel_userinput(user_question)



def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text

def get_pdf_text(pdf):
    pdf_reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_csv_text(file):
    return "a"

def get_text_chunks(text):
    # spilit ito chuncks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
    
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    return knowledge_base

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def get_conversation_chain(vetorestore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":5,
                                                      "max_length":64})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handel_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))
        st.write(f"Total Tokens: {cb.total_tokens}" f", Prompt Tokens: {cb.prompt_tokens}" f", Completion Tokens: {cb.completion_tokens}" f", Total Cost (USD): ${cb.total_cost}")


if __name__ == '__main__':
    main()