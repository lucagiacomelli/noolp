import nltk


class ParserBase:
    """Class responsible for parsing a document and extract relevant information."""

    def __init__(self, document: str, language: str = "english", verbose=False):
        self.document = document
        self.language = language
        self.verbose = verbose

    def tokenize_document(self) -> list:
        return nltk.word_tokenize(self.document)

    def part_of_speech_tags(self) -> list:
        tokens = self.tokenize_document()
        return nltk.pos_tag(tokens)
