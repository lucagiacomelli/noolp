import string
from typing import List

from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

from noolp.constants import Constants
from noolp.parser.exceptions import ParserLanguageException


class ParserBase:
    """
    Class responsible for parsing a document and extract relevant information.
    The document can be a story (list of sentences) or a simple sentence.

    """

    def __init__(self, document: str, language: str = "english", verbose: bool = False):
        self.document = document
        self.language = language
        self.verbose = verbose

    def _punctuation_set(self) -> set:
        punctuation_set = set(string.punctuation)
        punctuation_set.update(["''", "``", "'s"])
        return punctuation_set

    def extract_sentences(self) -> List[str]:
        """Extract sentences from a given document"""

        sentences = nltk.sent_tokenize(self.document)
        return sentences

    def tokenize(
        self, include_stop_words: bool = True, include_punctuation: bool = True
    ) -> List[List[str]]:
        """
        Extracts the sentences from the document and, for each sentence, extract its tokens.

        :param include_stop_words:
        :param include_punctuation:
        :return:
        """

        stop_words_set = set(stopwords.words(self.language))
        punctuation_set = self._punctuation_set()
        sentences = self.extract_sentences()
        tokens_sentences = []
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)

            if not include_stop_words:
                tokens = [token for token in tokens if token not in stop_words_set]
            if not include_punctuation:
                tokens = [token for token in tokens if token not in punctuation_set]
            tokens_sentences.append(tokens)

        return tokens_sentences

    def part_of_speech_tags(
        self, include_stop_words: bool = True, include_punctuation: bool = True
    ) -> List[List[tuple]]:
        """
        Returns the Part of Speech tags for every sentence in the document.
        In order to extracts the tags the sentences are tokenized (considering punctuation and stop words).

        :param include_stop_words:
        :param include_punctuation:
        :return:
        """

        stop_words_set = set(stopwords.words(self.language))
        punctuation_set = self._punctuation_set()
        tokens_sentences = self.tokenize()
        pos_tags_sentences = []
        for tokens_sentence in tokens_sentences:
            pos_tags = nltk.pos_tag(tokens_sentence)
            if not include_stop_words:
                pos_tags = [
                    pos_tag for pos_tag in pos_tags if pos_tag[0] not in stop_words_set
                ]
            if not include_punctuation:
                pos_tags = [
                    pos_tag for pos_tag in pos_tags if pos_tag[0] not in punctuation_set
                ]
            pos_tags_sentences.append(pos_tags)

        return pos_tags_sentences

    def _get_pos_tag_for_lemmatization(self, pos: str) -> str:
        if pos.startswith("NN"):
            return "n"
        elif pos.startswith("VB"):
            return "v"
        elif pos.startswith("JJ"):
            return "a"
        else:
            return "n"

    def lemmatize(
        self,
        include_stop_words: bool = True,
        include_punctuation: bool = True,
        include_reporting_verbs: bool = True,
    ) -> List[List[str]]:
        """
        Returns the lemmas of each sentence of the document.
        The lemmas are calculating from the part of speech tags.

        :param include_stop_words: if True, include the stop words in the returned lemmas
        :param include_punctuation: if True, include the stop words in the returned lemmas
        :param include_reporting_verbs: if True, include the reporting verbs in the returned sentences
        :return:
        """
        lemmatizer = WordNetLemmatizer()

        pos_tags_sentences = self.part_of_speech_tags(
            include_stop_words=include_stop_words,
            include_punctuation=include_punctuation,
        )
        lemmas_sentences = []
        for pos_tags_sentence in pos_tags_sentences:
            list_lemmas = [
                lemmatizer.lemmatize(word, pos=self._get_pos_tag_for_lemmatization(pos))
                for word, pos in pos_tags_sentence
            ]

            if not include_reporting_verbs:
                try:
                    list_lemmas = [
                        lemma
                        for lemma in list_lemmas
                        if lemma not in Constants.reporting_verbs[self.language]
                    ]
                except KeyError:
                    raise ParserLanguageException(
                        f"Reporting verbs for language {self.language} are not available."
                    )

            lemmas_sentences.append(list_lemmas)
        return lemmas_sentences

    def extract_lemmatized_sentences(
        self, include_stop_words: bool = True, include_punctuation: bool = True
    ) -> List[str]:
        """
        Extract sentences with only lemmas from a given document.

        :param include_stop_words: if True, include the stop words in the returned sentences
        :param include_punctuation: if True, include punctuation in the returned sentences
        """

        lemmas_sentences = self.lemmatize(
            include_stop_words=include_stop_words,
            include_punctuation=include_punctuation,
        )
        sentences = []
        for lemmas_sentence in lemmas_sentences:
            sentence_with_only_lemmas = " ".join(lemmas_sentence)
            sentences.append(sentence_with_only_lemmas)
        return sentences

    def most_common_terms(
        self, include_stop_words: bool = True, include_punctuation: bool = True
    ):
        lemmas_sentences = self.lemmatize(
            include_stop_words=include_stop_words,
            include_punctuation=include_punctuation,
        )
        dictionary = corpora.Dictionary(lemmas_sentences)
        return dictionary.most_common()
