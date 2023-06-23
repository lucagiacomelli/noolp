from typing import List

from noolp.parser.tfidf_parser import TfdifParser
from noolp.topic_modelling.topic_modeller import TopicModeller


class LDATopicModeller(TopicModeller):
    def __init__(
        self,
        name,
        document,
        number_topics=3,
        number_passes=20,
        words_per_topic=3,
        verbose: bool = False,
    ):
        super().__init__(name, document, verbose)

        self.number_topics = number_topics
        self.number_passes = number_passes
        self.words_per_topic = words_per_topic

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

    def get_lda_topics(self, documents: List[List[str]]):

        import gensim
        from gensim import corpora

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

        # coherence_model_lda = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='c_v')
        # coherence_lda = coherence_model_lda.get_coherence()
        # print('\nCoherence Score: ', coherence_lda)

        return lda_model.show_topics(
            num_topics=self.number_topics,
            num_words=self.words_per_topic,
            formatted=False,
        )

    def extract_topics(self, use_lemmatization: bool = True):
        story = self._clean_story(self.document)

        tfidf_parser = TfdifParser(document=story, verbose=True)

        lemmas = tfidf_parser.lemmatize(
            include_stop_words=False, include_punctuation=False
        )
        topics = self.get_lda_topics(lemmas)
        return topics
