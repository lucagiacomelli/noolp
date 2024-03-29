from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np

from noolp.document_similarity.doc_similarity import DocSimilarity
from noolp.parser.tfidf_parser import TfdifParser


class TFIDFSimilarity(DocSimilarity):
    """
    Two documents are similar if they contain the same terms,
    which are not repetitive in the entire corpus of documents.
    So the similarity of two documents is also affected by the other documents in the corpus.

    TF (Term Frequency): raw count of occurrences divided by number of words in the document
    IDF (Inverse Document Frequency): calculate how common a word is amongst the corpus.
        log(Number of documents/number of documents with that word)

    """

    def __init__(
        self,
        documents: List[str],
        language: str = "english",
        verbose=False,
        norm="l2",
        metric="cosine",
    ):
        super().__init__(documents, language, verbose)
        self.norm = norm
        self.metric = metric

    def get_vectors(self):
        clean_documents = [
            TfdifParser(document=document).clean_document()
            for document in self.documents
        ]

        # L2 normalization by default
        vectorizer = TfidfVectorizer(norm=self.norm)
        tfidf_vectors = vectorizer.fit_transform(clean_documents)
        feature_names = vectorizer.get_feature_names_out()

        if self.verbose:
            print(clean_documents)
            print(feature_names)
            print(tfidf_vectors.toarray())
            print(tfidf_vectors.T.toarray())

        return tfidf_vectors

    def get_similarity(self) -> List[List[float]]:
        """
        Calculate the similarity considering different distances between vectors of frequencies.
        The main limitation of the TF-IDF similarity is Matrix Sparsity and the actual sense of each word.
        """

        tfidf_similarities = []
        tfidf_vectors = self.get_vectors()

        # compute the cosine similarity between the TF-IDF vectors.
        # The normalization is already done during the TF-iDF vectors extraction
        if self.metric == "cosine":
            tfidf_similarities = np.dot(tfidf_vectors, tfidf_vectors.T).toarray()
        if self.metric == "euclidean":
            tfidf_similarities = 1 - euclidean_distances(tfidf_vectors)

        return tfidf_similarities
