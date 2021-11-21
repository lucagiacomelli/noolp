import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from constants import Constants


class ParserBase:
    """
    Class responsible for parsing a document and extract relevant information.
    The document can be a story (list of sentences) or a simple sentence.

    """

    def __init__(self, document: str, language: str = "english", verbose=False):
        self.document = document
        self.language = language
        self.verbose = verbose

    def punctuation_set(self) -> set:
        punctuation_set = set(string.punctuation)
        punctuation_set.update(["''", "``", "'s"])
        return punctuation_set

    def extract_sentences(self, story: str):
        """
        Extract sentence from a given story

        """
        sentences = nltk.sent_tokenize(story)
        return sentences

    def tokenize(self, include_stop_words: bool = True, include_punctuation: bool = True) -> list:
        """"""

        sentences = self.extract_sentences(self.document)
        tokens_sentences = []
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            stop_words_set = set(stopwords.words(self.language))
            punctuation_set = self.punctuation_set()

            if not include_stop_words:
                tokens = [token for token in tokens if token not in stop_words_set]
            if not include_punctuation:
                tokens = [token for token in tokens if token not in punctuation_set]
            tokens_sentences.append(tokens)

        return tokens_sentences

    def part_of_speech_tags(self, include_stop_words: bool = True, include_punctuation: bool = True) -> list:
        """"""

        tokens_sentences = self.tokenize()
        pos_tags_sentences = []
        for tokens_sentence in tokens_sentences:
            stop_words_set = set(stopwords.words(self.language))
            punctuation_set = self.punctuation_set()

            pos_tags = nltk.pos_tag(tokens_sentence)
            if not include_stop_words:
                pos_tags = [pos_tag for pos_tag in pos_tags if pos_tag[0] not in stop_words_set]
            if not include_punctuation:
                pos_tags = [pos_tag for pos_tag in pos_tags if pos_tag[0] not in punctuation_set]
            pos_tags_sentences.append(pos_tags)

        return pos_tags_sentences

    def _get_pos_tag_for_lemmatization(self, POS):
        if POS.startswith("NN"):
            return "n"
        elif POS.startswith("VB"):
            return "v"
        elif POS.startswith("JJ"):
            return "a"
        else:
            return "n"

    def lemmatize(self, include_stop_words: bool = True, include_punctuation: bool = True, include_reporting_verbs: bool = True) -> list:
        lemmatizer = WordNetLemmatizer()

        pos_tags_sentences = self.part_of_speech_tags(include_stop_words=include_stop_words, include_punctuation=include_punctuation)
        lemmas_sentences = []
        for pos_tags_sentence in pos_tags_sentences:
            list_lemmas = [lemmatizer.lemmatize(word, pos=self._get_pos_tag_for_lemmatization(pos)) for word, pos in pos_tags_sentence]

            if not include_reporting_verbs:
                list_lemmas = [lemma for lemma in list_lemmas if lemma not in Constants.reporting_verbs]

            lemmas_sentences.append(list_lemmas)

        return lemmas_sentences
