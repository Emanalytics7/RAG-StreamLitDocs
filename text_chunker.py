from langchain.text_splitter import CharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=0):
        self.text_splitter = CharacterTextSplitter(
                                                   chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=len
                                                   )
        
    def chunk_text(self, document):
        return self.text_splitter.create_documents(document)
