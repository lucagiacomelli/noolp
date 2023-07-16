from pytest import fixture

from noolp.document_similarity.doc_similarity import DocSimilarity


class TestDocSimilarity:

    doc1 = "This is a function to test document_path_similarity."
    doc2 = "Use this function to see if your code in doc_to_synsets and similarity_score is correct!"

    @fixture
    def instance(self):
        return DocSimilarity(documents=[self.doc1, self.doc2])
