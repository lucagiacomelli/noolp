from typing import List

import numpy as np

from noolp.document_similarity.doc_similarity import DocSimilarity
from noolp.parser import ParserBase


class JaccardSimilarity(DocSimilarity):
    def __init__(self, documents: List[str], language: str = "english", verbose=False):
        super().__init__(documents, language, verbose)

    def get_similarity(self) -> List[List[float]]:
        """"""

        jaccard_similarities = self.init_matrix_similarities()
        for (i1, doc1), (i2, doc2) in self.pairs_of_documents():

            parser = ParserBase(document=doc1)
            parser2 = ParserBase(document=doc2)

            # consider all the lemmas of all the sentences in each parsed document
            words_document1 = set().union(*parser.lemmatize(include_stop_words=False, include_punctuation=False))
            words_document2 = set().union(*parser2.lemmatize(include_stop_words=False, include_punctuation=False))

            intersection = words_document1.intersection(words_document2)
            union = words_document1.union(words_document2)

            jaccard = float(len(intersection)) / len(union)

            if self.verbose:
                print(f"\nJaccard Similarity between documents {i1} and {i2}: {jaccard}")

            jaccard_similarities[i1][i2] = jaccard

        return np.array(jaccard_similarities)
