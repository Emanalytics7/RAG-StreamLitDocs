import os 
from docx import Document
import PyPDF2

class DocumentProcessor:
    @staticmethod
    def get_text(uploaded_file):
        file_extention = os.path.splittext(uploaded_file.name)[1].lower()
        if file_extention == '.pdf':
            return DocumentProcessor.get_pdf_text(uploaded_file)
        elif file_extention == '.docx':
            return DocumentProcessor.get_docx_text(uploaded_file)
        else:
            return 'Unsupported File Type!'
        
    @staticmethod
    def get_pdf_text(pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text
   
    @staticmethod
    def get_docx_text(docx_file):
        doc = Document(docx_file)
        text = ' '.join([paragraph.text for paragraph in doc.paragraph])
        return text
    
        
        
