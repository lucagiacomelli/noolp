import nltk


class DocSimilarity:
    """
    Class responsible for finding the document similarity between two documents.

    """

    def __init__(self, document1: str, document2: str, language: str = "english", verbose=False):
        self.doc1 = document1
        self.doc2 = document2
        self.language = language
        self.verbose = verbose

    def tokenize_document(self, document: str) -> list:
        return nltk.word_tokenize(document)

    def part_of_speech_tags(self, document: str) -> list:
        tokens = self.tokenize_document(document=document)
        return nltk.pos_tag(tokens)

    def get_similarity(self):
        """Get the similarity value between the documents"""
        pass
