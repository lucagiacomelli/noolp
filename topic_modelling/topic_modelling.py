import string
import typing

import gensim
import nltk
from gensim import corpora
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from parser.parser_base import ParserBase


class TopicModeller:
    def __init__(self, name, number_topics=3, number_passes=50, words_per_topic=3):
        self.name = name
        self.number_topics = number_topics
        self.number_passes = number_passes
        self.words_per_topic = words_per_topic

    def clean_story(self, story):
        # story = story.decode('utf8')

        # Handle the case with .I, .You, .It, .He, .She, .We, .They
        story = story.replace(".I ", ". I ")
        story = story.replace(".You ", ". You ")
        story = story.replace(".He ", ". He ")
        story = story.replace(".She ", ". she ")
        story = story.replace(".It ", ". It ")
        story = story.replace(".We ", ". We ")
        story = story.replace(".They ", ". They ")

        # Remove the 'Read More' part from the story
        if "Read more" in story:
            index_read_more = story.index("Read more")
            story = story[:index_read_more]

        return story

    """
    Extract tokens from a sentence,
    removing English stopwords and punctuations
    @Return a list of tokens (strings)
    """

    def get_tokens(self, sentence):
        stop = set(stopwords.words("english"))
        punctuation = set(string.punctuation)
        punctuation.update(["''", "``", "'s"])

        tokens = nltk.word_tokenize(sentence)
        tokens_pos = nltk.pos_tag(tokens)

        stop_free = [i for i in tokens_pos if i[0] not in stop]
        punc_free = [ch for ch in stop_free if ch[0] not in punctuation]
        return punc_free

    def extract_sentences(self, story):
        """
        Extract sentence from a given story
        :param story:

        """
        sentences = nltk.sent_tokenize(story)
        return sentences

    def lemmatize_story(self, story):
        """

        :return a list of cleaned and lemmatized documents (sentences - strings)
        """

        story = self.clean_story(story)
        normalized_sentences = []

        parser = ParserBase(document=story)
        lemmas_sentences = parser.lemmatize(include_stop_words=False, include_punctuation=False)

        for lemmas_sentence in lemmas_sentences:
            sentence_with_only_lemmas = " ".join(lemmas_sentence)
            normalized_sentences.append(sentence_with_only_lemmas)

        return normalized_sentences

    def tdf_idf_vectorize(self, sentences: list) -> typing.List[list]:
        """Retturns the TDF-IDF vectors from the sentences"""
        vectorizer = TfidfVectorizer()
        tdf_idf_vectors = vectorizer.fit_transform(sentences)

        print("TDIDF vectors")
        print(tdf_idf_vectors.shape)
        feature_names = vectorizer.get_feature_names_out()
        print(len(feature_names))
        print(feature_names)
        return tdf_idf_vectors

    """
    The core idea of the Latent Semantic Analysis is to generate a document-term matrix
    and decompose it into two matrices: document-topic and topic-term.
    Document-term matrix: how many times the term j appears in the document i.
    Because this matrix is sparse, we need a dimensionality reduction on the matrix. We do that
    with the Singular Value Decomposition (SVD)
    """

    def get_LSA_topics(self, documents):

        # raw documents to tf-idf matrix
        vectorizer = TfidfVectorizer(stop_words="english", use_idf=True, smooth_idf=True)

        svd_model = TruncatedSVD(n_components=100, algorithm="randomized", n_iter=10)

        svd_transformer = Pipeline([("tfidf", vectorizer), ("svd", svd_model)])
        svd_matrix = svd_transformer.fit_transform(documents)

        # use the matrix to find topics and similarities
        return

    """
       Extract topics from a story using
       Latent Dirichlet Allocation. We calculate the document-term matrix
       and apply the LDA model.
       @Return a list of tuples where erach tuple has a set of relevant words for the topic
       """

    def get_LDA_topics(self, documents):

        # Creating the term dictionary of our courpus, where every unique term is assigned an index.
        dictionary = corpora.Dictionary(documents)

        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]

        # Creating the object for LDA model
        Lda = gensim.models.ldamodel.LdaModel

        # Running and Trainign LDA model on the document term matrix.
        # print len(documents)

        # chunksize = len(documents)/5
        ldamodel = Lda(doc_term_matrix, num_topics=self.number_topics, id2word=dictionary, passes=self.number_passes)

        return ldamodel.show_topics(num_topics=self.number_topics, num_words=self.words_per_topic, formatted=False)

    def extract_topics(self, story, use_lemmatization: bool = True):

        if use_lemmatization:
            sentences = self.lemmatize_story(story)
        else:
            sentences = ParserBase(document=story).extract_sentences()

        tdf_idf_vectors = self.tdf_idf_vectorize(sentences=sentences)

        # topics = self.get_LDA_topics(lemmatized_sentences)

        # sentences = self.extract_sentences(story)
        # print(sentences)
        # lsa_topics = self.get_LSA_topics(sentences)

        return []
        # return topics
