import os
import logging
from docx import Document
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

# configure logging
logging.basicConfig(level=logging.INFO)

class DocumentProcessor:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
    
    def get_text(self, uploaded_file):
        if not uploaded_file:
            logging.error('No file provided.')
            raise ValueError('File is empty or not provided!')

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension in ['.pdf', '.docx']:
            return self._extract_text(uploaded_file, file_extension)
        else:
            logging.error(f'Unsupported file type: {file_extension}')
            raise ValueError('Unsupported File Type!')

    def _extract_text(self, uploaded_file, file_extension):
        try:
            if file_extension == '.pdf':
                return self._get_pdf_text(uploaded_file)
            elif file_extension == '.docx':
                return self._get_docx_text(uploaded_file)
        except Exception as e:
            logging.exception(f'Error processing file: {e}')
            raise

    def _get_pdf_text(self, pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''.join([page.extract_text() or '' for page in reader.pages])
        return text

    def _get_docx_text(self, docx_file):
        doc = Document(docx_file)
        return ' '.join(paragraph.text for paragraph in doc.paragraphs)

    def generate_response(self, uploaded_file, query_text):
        document_text = self.get_text(uploaded_file)
        if document_text:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.create_documents([document_text])
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            db = Chroma.from_documents(texts, embeddings)
            retriever = db.as_retriever()
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=self.openai_api_key),
                                              chain_type='stuff', retriever=retriever)
            return qa.run(query_text)
        else:
            raise ValueError("Failed to extract text from document.")
