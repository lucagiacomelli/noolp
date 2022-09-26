from typing import List

import gensim
from gensim import corpora

# from gensim.models.coherencemodel import CoherenceModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from noolp.parser.tfidf_parser import TfdifParser


class TopicModeller:
    def __init__(self, name, number_topics=3, number_passes=20, words_per_topic=3, verbose: bool = False):
        self.name = name
        self.number_topics = number_topics
        self.number_passes = number_passes
        self.words_per_topic = words_per_topic
        self.verbose = verbose

    def _clean_story(self, story: str) -> str:

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

    def get_LDA_topics(self, documents: List[List[str]]):

        dictionary = corpora.Dictionary(documents)

        # Converting list of documents (corpus) into Document Term Frequency Matrix.
        # Each entry (j, f) in the row i is a tuple that describes the frequency f of the word j in the document i
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]

        # NOTE: we can also save the trained model and load the model for unseen documents

        # number of documents to be used in each training chunk.
        chunksize = len(documents) / 3
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=doc_term_matrix,
            id2word=dictionary,
            num_topics=self.number_topics,
            random_state=100,
            update_every=1,
            chunksize=chunksize,
            passes=self.number_passes,
            alpha="auto",
            per_word_topics=True,
        )

        if self.verbose:
            print([[(dictionary[id], freq) for id, freq in cp] for cp in (doc_term_entry for doc_term_entry in doc_term_matrix)])

            print(lda_model.print_topics())
            print("\nPerplexity: ", lda_model.log_perplexity(doc_term_matrix))

        # coherence_model_lda = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='c_v')
        # coherence_lda = coherence_model_lda.get_coherence()
        # print('\nCoherence Score: ', coherence_lda)

        return lda_model.show_topics(num_topics=self.number_topics, num_words=self.words_per_topic, formatted=False)

    def extract_topics(self, story, use_lemmatization: bool = True):
        story = self._clean_story(story)

        tfidf_parser = TfdifParser(document=story, verbose=True)

        lemmas = tfidf_parser.lemmatize(include_stop_words=False, include_punctuation=False)
        topics = self.get_LDA_topics(lemmas)
        # lsa_topics = self.get_LSA_topics(sentences)

        return topics
