from noolp.parser.parser_base import ParserBase


class TfdifParser(ParserBase):
    """ """

    def __init__(self, document: str, language: str = "english", verbose=False):
        super().__init__(document, language, verbose)

    def clean_document(self):
        """Returns the TDF-IDF vectors from the sentences of the document."""

        sentences = self.extract_lemmatized_sentences(
            include_punctuation=False, include_stop_words=False
        )
        cleaned_document = ". ".join(sentences)

        return cleaned_document
