from typing import List


class DocSimilarity:
    """
    Abstract Class responsible for finding the document similarity between multiple documents.
    """

    def __init__(self, documents: List[str], language: str = "english", verbose=False):
        self.documents = documents
        self.language = language
        self.verbose = verbose

        if len(self.documents) < 2:
            raise AttributeError(
                "The list of documents should contain at list two strings."
            )

    @property
    def n(self) -> int:
        return len(self.documents)

    def init_matrix_similarities(self) -> List[List[int]]:
        matrix_similarities = [[0] * self.n for _ in range(self.n)]

        # the documents are identical to themselves
        for i in range(self.n):
            matrix_similarities[i][i] = 1
        return matrix_similarities

    def pairs_of_documents(self):
        result = []
        for i1 in range(self.n):
            for i2 in range(i1 + 1, self.n):
                result.append(((i1, self.documents[i1]), (i2, self.documents[i2])))
        return result

    def get_similarity(self) -> List[List[float]]:
        """
        Calculate the actual similarity matrix value between the documents.
        This function is overwritten
        """
        pass

    def most_similar_pairs(self, index_top_pairs: int = 3) -> list:
        """
        Return the indexes and the score of the most similar pair
        among all the list of documents.
        """

        similarity_matrix = self.get_similarity()
        pairs = []
        for i in range(len(similarity_matrix) - 1):
            for j in range(i + 1, len(similarity_matrix)):
                pairs.append({"index": [i, j], "score": similarity_matrix[i][j]})

        # Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)
        return pairs[:index_top_pairs]

    def most_similar_pair(self) -> dict:
        """
        Return the indexes and the score of the most similar pair
        among all the list of documents.
        """
        return self.most_similar_pairs()[0]
