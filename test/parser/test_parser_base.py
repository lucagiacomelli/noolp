from pytest import fixture

from parser.parser_base import ParserBase


class TestParserBase:

    doc1 = "This is a function to test document path similarity."
    doc2 = "Use this function to see if your code in doc_to_synsets and similarity_score is correct!"

    @fixture
    def instance(self):
        return ParserBase(document=self.doc1)

    def test_tokenize(self, instance):
        assert instance.tokenize() == ["This", "is", "a", "function", "to", "test", "document", "path", "similarity", "."]
        assert instance.tokenize(include_stop_words=False) == ["This", "function", "test", "document", "path", "similarity", "."]
        assert instance.tokenize(include_stop_words=False, include_punctuation=False) == [
            "This",
            "function",
            "test",
            "document",
            "path",
            "similarity",
        ]
