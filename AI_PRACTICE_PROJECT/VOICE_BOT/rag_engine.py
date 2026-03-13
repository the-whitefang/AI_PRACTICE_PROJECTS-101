from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class RAGEngine:

    def __init__(self):

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.documents = []
        self.index = None

        self.load_documents()

    def load_documents(self):

        with open("knowledge_base/friday_docs.txt") as f:
            docs = f.readlines()

        self.documents = [d.strip() for d in docs if d.strip()]

        embeddings = self.embedder.encode(self.documents)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(np.array(embeddings))

    def search(self, query):

        query_embedding = self.embedder.encode([query])

        distances, indices = self.index.search(query_embedding, 3)

        results = [self.documents[i] for i in indices[0]]

        return "\n".join(results)