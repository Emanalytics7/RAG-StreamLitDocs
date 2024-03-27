import os 
from docx import Document
import PyPDF2

class DocumentProcessor:
    @staticmethod
    def get_text(uploaded_file):
        file_extention = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extention == '.pdf':
            return DocumentProcessor.get_pdf_text(uploaded_file)
        elif file_extention == '.docx':
            return DocumentProcessor.get_docx_text(uploaded_file)
        else:
            return 'Unsupported File Type!'
        
    @staticmethod
    def get_pdf_text(pdf_file):
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
            return text
        except Exception as e:
            return f'Error reading PDF: {str(e)}'
   
    @staticmethod
    def get_docx_text(docx_file):
        try:
            doc = Document(docx_file)
            text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            return f'Error reading DOCX: {str(e)}'
    
        
# Example for Testing Document Processor
# processor = DocumentProcessor()
# text_from_pdf = processor.get_pdf_text('sample-pdf-file.pdf')
# text_from_docx = processor.get_docx_text('sample-docx-file-for-testing.docx')
# print(text_from_pdf)
# print(text_from_docx)





