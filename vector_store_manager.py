from langchain_community.vectorstores import chroma
import chromadb
import openai
class VectorStoreManager:
    def __init__(self, collection_name, openai_api_key):
        self.chroma_client = chromadb.PersistentClient('./chroma_storage')
        self.collection = self.chroma_client.create_collection(name=collection_name)
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key

    @staticmethod
    def get_openai_embeddings(text_chunks, openai_api_key):
        openai_api_key = openai_api_key
        embeddings = []
        for chunk in text_chunks:
            response = openai.embeddings.create(
                input=chunk,
                model='text-embedding-3-large'
            ).data[0].embedding
            embeddings.append(response)
        return embeddings


    def store_embeddings(self, text_chunks):
        embeddings = self.get_openai_embeddings(text_chunks, self.openai_api_key)
        for chunk, embeddings in zip(text_chunks, embeddings):
            chunk_str = str(chunk)
            self.collection.add(documents=[embeddings], ids=[chunk_str])
 

    def query_embeddings(self, query_text, n_results=2):
        query_embedding = self.get_openai_embeddings([query_text], self.openai_api_key)[0]
        results = self.collection.query(query_texts=[query_embedding], n_results=n_results)
        return results

