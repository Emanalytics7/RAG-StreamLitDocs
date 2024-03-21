import chromadb

class VectorStoreManager:
    def __init__(self, collection_name):
        self.chroma_client = chromadb.PersistentClient('./chroma_storage')
        self.collection = self.chroma_client.create_collection(name=collection_name)

    def store_embeddings(self, text_chunks, embeddings_model):
        for chunk in text_chunks:
            embedding = embeddings_model.encode(chunk)
            self.collection.add(documents=[embedding], ids=[chunk])

    def query_embeddings(self, query_embedding, n_results=2):
        return self.collection.query(query_texts=[query_embedding], n_results=n_results)
