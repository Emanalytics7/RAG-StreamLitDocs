from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import fitz  
from docx import Document
import io
import magic

class DocumentProcessor:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def read_pdf(self, file_content):
        text = []
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)

    def read_docx(self, file_content):
        doc = Document(io.BytesIO(file_content))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def process_file(self, uploaded_file):
        mime_type = magic.from_buffer(uploaded_file.read(1024), mime=True)
        uploaded_file.seek(0)  # reset file pointer after reading

        if mime_type == 'application/pdf':
            document_text = self.read_pdf(uploaded_file)
        elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                            'application/msword']:
            uploaded_file.seek(0)
            document_text = self.read_docx(uploaded_file.read())
        elif mime_type == 'text/plain':
            document_text = uploaded_file.read().decode('utf-8')
        else:
            raise ValueError("Unsupported file type")

        return document_text

    def generate_response(self, uploaded_file, query_text):
        document_text = self.process_file(uploaded_file)

        if document_text:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.create_documents([document_text])
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            db = Chroma.from_documents(texts, embeddings)
            retriever = db.as_retriever()
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=self.openai_api_key),
                                              chain_type='stuff', retriever=retriever)
            return qa.run(query_text)
