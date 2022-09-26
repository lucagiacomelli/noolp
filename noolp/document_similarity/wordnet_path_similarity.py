import typing

from nltk.corpus import wordnet as wn
import numpy as np

from noolp.document_similarity.doc_similarity import DocSimilarity
from noolp.parser import ParserBase


class WordNetPathSimilarity(DocSimilarity):
    """
    The distance measure is the length of the shortest path in the WordNet graph that
    divides the different synset of the documents.

    The similarity of two documents is not symmetric.
    """

    def __init__(self, documents: typing.List[str], language: str = "english", verbose=False):
        super().__init__(documents, language, verbose)

    def _convert_tag(self, wordnet_tag) -> typing.Optional[str]:
        """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""

        tag_dict = {"N": "n", "J": "a", "R": "r", "V": "v"}
        try:
            return tag_dict[wordnet_tag[0]]
        except KeyError:
            return None

    def doc_to_synsets(self, document: str, synset_per_word: int = 1, include_stop_words: bool = False, include_punctuation: bool = False) -> list:
        """
        Returns a list of WordNet synsets in document after extracting the Parts Of Speech from it.

        :param document: string to be converted
        :param synset_per_word: number of synset per a given word to consider in the similarity
        :param include_stop_words: if True, include the stop words in the extraction of the POS tags
        :param include_punctuation: if True, include the punctuation in the extraction of the POS tags

        :return list of synsets [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]

        """

        parser = ParserBase(document=document)
        pos_tags_sentences = parser.part_of_speech_tags(include_stop_words=include_stop_words, include_punctuation=include_punctuation)

        synsets: list = []
        for pos_tags_sentence in pos_tags_sentences:
            for word, pos in pos_tags_sentence:
                try:
                    all_synsets = wn.synsets(word, self._convert_tag(pos))
                    if len(all_synsets) > 0:
                        synsets.extend(all_synsets[0:synset_per_word])
                except Exception:
                    continue

        return synsets

    def similarity_score(self, synset_list_1: list, synset_list_2: list):
        """
        Calculate the normalized similarity score of s1 onto s2

        For each synset in s1, finds the synset in s2 with the largest similarity value.
        Sum of all of the largest similarity values and normalize this value by dividing it by the
        number of largest similarity values found.

        :param synset_list_1
        :param synset_list_2

        """

        largest_similarity_values = []
        for syn in synset_list_1:
            max_similarity = 0.0
            for syn2 in synset_list_2:
                path_similarity = syn.path_similarity(syn2)
                if self.verbose:
                    print(f"Path similarity between {syn} and {syn2}: {path_similarity}")
                if syn.path_similarity(syn2) is not None and syn.path_similarity(syn2) > max_similarity:
                    max_similarity = syn.path_similarity(syn2)

            if max_similarity != 0.0:
                largest_similarity_values.append(max_similarity)

        if not largest_similarity_values:
            return 0

        return sum(largest_similarity_values) / len(largest_similarity_values)

    def get_similarity(self) -> typing.List[typing.List[float]]:
        """
        Path similarity is a similarity measure that finds the distance that is the length of the shortest path between two synsets
        using WordNet. The similarity of the documents is the average of the length two paths.

        :return the matrix of the similarities of each pair of documents.

        """

        # extract Wordnet synset from documents
        synsets_list = []
        for document in self.documents:
            synsets_list.append(self.doc_to_synsets(document))

        path_similarities = self.init_matrix_similarities()
        for (i1, doc1), (i2, doc2) in self.pairs_of_documents():
            synsets1 = synsets_list[i1]
            synsets2 = synsets_list[i2]

            path_similarity = (self.similarity_score(synsets1, synsets2) + self.similarity_score(synsets2, synsets1)) / 2

            if self.verbose:
                print(f"\nPath Similarity between the documents {i1} and {i2}: {path_similarity}")

            path_similarities[i1][i2] = path_similarity

        return np.array(path_similarities)
