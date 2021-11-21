from pytest import fixture

from document_similarity.doc_similarity import DocSimilarity


class TestDocSimilarity:

    doc1 = "This is a function to test document_path_similarity."
    doc2 = "Use this function to see if your code in doc_to_synsets and similarity_score is correct!"

    @fixture
    def instance(self):
        return DocSimilarity(document1=self.doc1, document2=self.doc2)


    def test_tokenize(self, instance):
        assert instance.tokenize_document(document=doc1)

