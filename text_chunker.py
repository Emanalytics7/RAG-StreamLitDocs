from langchain.text_splitter import RecursiveCharacterTextSplitter
from document_processor import DocumentProcessor
import logging

logging.basicConfig(level=logging.INFO)

class TextChunker:
    """
    A class for chunking text documents based on character count,
    with optional overlap between chunks.
    
    """
    
    def __init__(self, separator='\n', chunk_size=1000, chunk_overlap=20):
        self.text_splitter = RecursiveCharacterTextSplitter(
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    length_function=len
                                                    )
        
    def chunk_text(self, document):
        """
        Splits the input document into chunks.
        
        Args:
            document (str): The text document to be chunked.
            
        Returns:
            list: A list of text chunks.
        """
        if not document:
            logging.warning('Document is empty or None.')
            return []
        try:
            return self.text_splitter.split_text(document)
        except Exception as e:
            logging.exception(f'Error while chunking text: {e}')
            return []





# Example for Testing TextChunker
# processor = DocumentProcessor()
# text_from_pdf = processor._get_pdf_text('sample-pdf-file.pdf')
# chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
# chunks = chunker.chunk_text(text_from_pdf)[1]
# print(chunks)