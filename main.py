from typing import List

from noolp.document_similarity.jaccard_similarity import JaccardSimilarity
from noolp.document_similarity.wordnet_path_similarity import WordNetPathSimilarity
from noolp.document_similarity.tfidf_similarity import TFIDFSimilarity
from noolp.topic_modelling.topic_modeller import TopicModeller

print("\nWelcome to NOOLP - A Natural Language Processing library!!\n")

doc1 = "This is a function to test document similarity."
doc2 = "Use this function to see if the similarity score is correct!"
doc3 = "Use this function to evaluate if the similarity score is right!"
story = (
    'A number of Santander cash machines have been shut down due to "potential criminal activity and vandalism", '
    'the bank has said. Lancashire Police Santander said "five ATMs were shut down immediately" but cash machines '
    'elsewhere remained "fully operational". Police in Cheshire said some ATMs were vandalised, but were not '
    'fitted with "skimming devices". People who have lost money have been urged to contact the bank and police. '
    'Lancashire Police said: "We are still taking reports but a series of similar incidents last week occurred in '
    'all parts of the county. The advice, therefore, is not limited to a specific area or town. "We are advising '
    "the public to be vigilant, in particular of Santander machines, but of any cash machines. Report anything "
    'suspicious, have a visual check of the cash point and if in doubt leave it and go somewhere else." Cheshire '
    'Police said on Monday: "We can confirm that there had been a number of ATMs tampered with over the weekend, '
    'however these are not thought to be linked to skimming devices. "It appeared that in one case keypad numbers '
    "had been stuck down in some way and a ruler had been attached to anther causing an issue with the buttons. "
    '"[The incidents] were confined to the eastern part of the county." Please include a contact number if you '
    "are willing to speak to a BBC journalist. You can also contact us in the following ways: Or use the form "
    "below If you are happy to be contacted by a BBC journalist please leave a telephone number that we can "
    "contact you on. In some cases a selection of your comments will be published, displaying your name as you "
    "provide it and location, unless you state otherwise. Your contact details will never be published. When "
    "sending us pictures, video or eyewitness accounts at no time should you endanger yourself or others, "
    "take any unnecessary risks or infringe any laws. Please ensure you have read the terms and conditions. "
)


class App:
    def get_similarities(self, documents: List[str]) -> dict:
        type_similarity_to_objects = {
            "chatgpt_similarity": WordNetPathSimilarity(
                documents=documents, verbose=False
            ),
            "wordnet_path_similarity": WordNetPathSimilarity(
                documents=documents, verbose=False
            ),
            "jaccard_similarity": JaccardSimilarity(documents=documents, verbose=False),
            "tfidf_cosine_similarity": TFIDFSimilarity(
                documents=documents, verbose=False, metric="cosine"
            ),
            "tfidf_euclidean_similarity": TFIDFSimilarity(
                documents=documents, verbose=False, metric="euclidean"
            ),
        }

        similarities = {}
        for similarity_type, similarity_obj in type_similarity_to_objects.items():
            similarities[similarity_type] = similarity_obj.get_similarity()

        for metric, similarities_matrix in similarities.items():
            print(f"\n{metric}:")
            print(similarities_matrix)

        return similarities


app = App()


documents = [doc1, doc2, doc3]

print("\n#### DOCUMENT SIMILARITY ####\n")
print(f"documents: {documents}")
doc_sim = app.get_similarities(documents=documents)
# doc_sim = app.get_similarities(documents=[doc1, doc2])


print("\n#### TOPIC EXTRACTION ####\n")
print(f"document: {story}")
topicModeller = TopicModeller("topic modeller 1")
print(f"Topics extracted: {topicModeller.extract_topics(story)}")
