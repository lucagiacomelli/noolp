import string

import nltk
from nltk.corpus import stopwords


class ParserBase:
    """Class responsible for parsing a document and extract relevant information."""

    def __init__(self, document: str, language: str = "english", verbose=False):
        self.document = document
        self.language = language
        self.verbose = verbose

    def tokenize_document(self, include_stop_words: bool = True, include_punctuation: bool = True) -> list:
        """"""

        tokens = nltk.word_tokenize(self.document)
        stop_words_set = set(stopwords.words(self.language))
        if not include_stop_words:
            tokens = [token for token in tokens if token not in stop_words_set]
        if not include_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]
        return tokens

    def part_of_speech_tags(self) -> list:
        tokens = self.tokenize_document()
        return nltk.pos_tag(tokens)
