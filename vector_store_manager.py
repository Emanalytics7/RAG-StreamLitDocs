import chromadb

class ChromaDBStorage:
    def __init__(self, db_path='./chroma_storage', collection_name='embedding6'):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(db_path)
        self.collection = self._get_or_create_collection(collection_name)

    def _get_or_create_collection(self, name):

        # check if the collection exists, and create it if it doesn't
        if name not in self.client.list_collections():
            return self.client.create_collection(name=name)
        return self.client.get_collection(name=name)

    def store_embedding(self, id, embedding):
        self.collection.add(ids=[id], embeddings=[embedding])

    def query_embedding(self, embedding, n_results=5):
        return self.collection.query(query_embeddings=[embedding], n_results=n_results)
