import nltk
from nltk.corpus import wordnet as wn


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""

    tag_dict = {"N": "n", "J": "a", "R": "r", "V": "v"}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


class DocSimilarity:
    """
    This class is responsible for finding the document similarity between two documents.

    """

    def __init__(self, document1: str, document2: str):
        self.doc1 = document1
        self.doc2 = document2

    def doc_to_synsets(self, document: str) -> list:
        """
        Returns a list of WordNet synsets in document.

        * Tokenize and tag the words in the document doc.
        * Find the first synset for each word/tag combination. If a synset is not found for that combination it is skipped.

        :param document: string to be converted
        :return list of synsets [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]

        """

        tokens = nltk.word_tokenize(document)
        tagger = nltk.pos_tag(tokens)
        synsets: list = []
        for word, pos in tagger:
            try:
                all_synsets = wn.synsets(word, convert_tag(pos))
                if len(all_synsets) > 0:
                    synsets.append(all_synsets[0])
            except Exception:
                continue
        return synsets

    def similarity_score(self, synset_list_1: list, synset_list_2: list):
        """
        Calculate the normalized similarity score of s1 onto s2

        For each synset in s1, finds the synset in s2 with the largest similarity value.
        Sum of all of the largest similarity values and normalize this value by dividing it by the
        number of largest similarity values found.

        """

        largest_similarity_values = []
        for syn in synset_list_1:
            max_similarity = 0.0
            for syn2 in synset_list_2:
                if syn.path_similarity(syn2) is not None and syn.path_similarity(syn2) > max_similarity:
                    max_similarity = syn.path_similarity(syn2)

            if max_similarity != 0.0:
                largest_similarity_values.append(max_similarity)

        sum = 0.0
        for value in largest_similarity_values:
            sum = sum + value

        return sum / len(largest_similarity_values)

    def document_path_similarity(self):
        """Finds the symmetrical similarity between doc1 and doc2"""

        synsets1 = self.doc_to_synsets(self.doc1)
        synsets2 = self.doc_to_synsets(self.doc2)

        print("Path Similarity between the documents: ")
        return (self.similarity_score(synsets1, synsets2) + self.similarity_score(synsets2, synsets1)) / 2
