from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, preprocessor=None, tokenizer=None):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer or preprocessor.tokenize
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []

    def build_index(self, documents):
        self.documents = documents
        self.tokenized_docs = [
            self.tokenizer(doc) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query, top_k=3):
        return [
            self.documents[i] for i, _ in self.search_with_scores(query, top_k)
        ]

    def search_with_scores(self, query, top_k=3):
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        return ranked[:top_k]
