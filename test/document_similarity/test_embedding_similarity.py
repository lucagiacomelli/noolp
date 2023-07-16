import numpy
from pytest import fixture

from noolp.document_similarity.embedding_similarity import EmbeddingSimilarity
from test.test_data import APPLICATIONS


class TestEmbeddingSimilarity:
    def test_most_similar_pair(self):
        application = APPLICATIONS[2]
        for question in application.get("questions"):
            new_documents = [question] + APPLICATIONS[4].get("questions")
            similarity_matrix = EmbeddingSimilarity(
                documents=new_documents, verbose=True
            ).get_similarity()

            similarities = similarity_matrix[0][1:]

            # sorted_indexes = numpy.argsort(-similarities)
            indexes = numpy.argpartition(similarities, -2)[-2:]
            sorted_indexes = indexes[numpy.argsort(-similarities[indexes])]
            top_scores = similarities[sorted_indexes]
            print(question)
            print(top_scores)
            print([new_documents[1:][i] for i in sorted_indexes])
            print()

            # indexes = most_similar_pair.get("index")
            # if 0 not in indexes:
            #     print(f"{new_documents[0]} - N/A\n")
            # else:
            #     similarity_dict = {
            #         f"{APPLICATIONS[0].get('program')}_question": new_documents[most_similar_pair.get("index")[0]],
            #         "dantia_question": new_documents[most_similar_pair.get("index")[1]],
            #         "score": most_similar_pair["score"],
            #     }
            #     print(similarity_dict)
            #     print()

        assert False
