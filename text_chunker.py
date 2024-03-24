from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, seperator='\n', chunk_size=900, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
                                                   chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=len
                                                   )
        
    def chunk_text(self, text):
        return self.text_splitter.split_text(text)
