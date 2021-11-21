from document_similarity.doc_similarity import DocSimilarity
from parser.parser_base import ParserBase


class JaccardSimilarity(DocSimilarity):
    def __init__(self, document1: str, document2: str):
        super().__init__(document1, document2)

    def get_similarity(self):
        """"""

        parser = ParserBase(document=self.doc1)
        parser2 = ParserBase(document=self.doc2)

        words_document1 = set().union(*parser.tokenize(include_stop_words=False, include_punctuation=False))
        words_document2 = set().union(*parser2.tokenize(include_stop_words=False, include_punctuation=False))

        intersection = words_document1.intersection(words_document2)
        union = words_document1.union(words_document2)

        jaccard = float(len(intersection)) / len(union)

        print(f"\nJaccard Similarity between the documents: {jaccard}")
        return jaccard
