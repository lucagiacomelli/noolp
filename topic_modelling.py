'''
Copyright 2017.
All rights reserved.

Topic modelling of a general story.

Author: luca.giacomelli@covatic.com (Luca Giacomelli)
'''
import string

import gensim
import nltk
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from constants import *


class TopicModeller:

    def __init__(self, documents):
        self.documents = documents

    def LDA_model(self):
        vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english',
                               token_pattern='(?u)\\b\\w\\w\\w+\\b')
        # Fit and transform
        X = vect.fit_transform(self.documents)

        print(X.shape)
        ## We have 2000 documents and 901 terms (tokens) that appear in at least 20
        ## documents and have at least 3 letters
        # print(X)

        # Convert sparse matrix to gensim corpus.
        corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

        # Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
        id_map = dict((v, k) for k, v in vect.vocabulary_.items())
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, id2word=id_map, num_topics=10, passes=25, random_state=34)

        return ldamodel

    def LDA_topics(ldamodel):
        result = []
        '''
        for i in range(10):
            topic = ldamodel.print_topic(i, topn=10)
            r = re.findall(r"\"(\d\d\d)", topic)
            for id in r:
                #print(r)
                topic = re.sub(id, id_map[int(id)], topic)
            #print(topic)
            result.append((i, topic))
        #print(result)
        '''
        #
        return ldamodel.print_topics(num_topics=10, num_words=10)

    def topic_distribution(doc):
        # Fit and transform

        X1 = vect.transform(doc)
        corpus2 = gensim.matutils.Sparse2Corpus(X1, documents_columns=False)
        ldamodel2 = gensim.models.ldamodel.LdaModel(corpus2, id2word=id_map, num_topics=10, passes=25, random_state=34)
        # id_map = dict((v, k) for k, v in vect.vocabulary_.items())
        # ldamodel.update(corpus2)
        list_of_list = list(ldamodel2.get_document_topics(corpus2))

        result = list_of_list[0]
        return result


'''
Class dedicated to Topic Modelling:
starting from a document (a story) we find a set of topics with related probabilities
'''


class TopicModeller2:

    def __init__(self, name, number_topics=3,
                 number_passes=50,
                 words_per_topic=3):
        self.name = name
        self.number_topics = number_topics
        self.number_passes = number_passes
        self.words_per_topic = words_per_topic

    '''
    Cleaning stories from BBC:
    - We remove pronouns
    - We remove the 'Read more' part at the end 
    @Return a string
    '''

    def clean_story(self, story):
        #story = story.decode('utf8')

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

    '''
    Extract tokens from a sentence,
    removing English stopwords and punctuations
    @Return a list of tokens (strings)
    '''
    def get_tokens(self, sentence):
        stop = set(stopwords.words('english'))
        punctuation = set(string.punctuation)
        punctuation.update(["''", "``", "'s"])

        tokens = nltk.word_tokenize(sentence)
        tokens_pos = nltk.pos_tag(tokens)

        stop_free = [i for i in tokens_pos if i[0] not in stop]
        punc_free = [ch for ch in stop_free if ch[0] not in punctuation]
        return punc_free

    '''
    Extract sentence from a given story
    @Return a list of sentences
    '''
    def extract_sentences(self, story):
        sentences = nltk.sent_tokenize(story)
        return sentences

    '''
    @Return a list of cleaned and lemmatized documents (sentences - strings)
    '''
    def lemmatize_story(self, story):
        story = self.clean_story(story)

        lemmatizer = WordNetLemmatizer()
        normalized_sentences = []
        for sentence in self.extract_sentences(story):
            tokens_pos = self.get_tokens(sentence)
            list_lemmas = [lemmatizer.lemmatize(word, pos=self.getPosTagForLemmatization(pos)) for word, pos in
                           tokens_pos]
            normalized_sentence = [lemma for lemma in list_lemmas if lemma not in Constants.reporting_verbs]
            normalized_sentences.append(normalized_sentence)

        # print "\nNormalized sentences: "
        # for norm_doc in normalized_sentences:
        #       print norm_doc

        return normalized_sentences

    '''
    Auxiliary method:
    convert the POS tag into the character as parameter for the lemmatization
    @Return a character
    '''
    def getPosTagForLemmatization(self, POS):
        if POS.startswith("NN"):
            return 'n'
        elif POS.startswith("VB"):
            return 'v'
        elif POS.startswith("JJ"):
            return 'a'
        else:
            return 'n'

    '''
    Extract topics from a story using
    Latent Dirichlet Allocation. We calculate the document-term matrix
    and apply the LDA model.
    @Return a list of tuples where erach tuple has a set of relevant words for the topic
    '''
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

    def extract_topics(self, story):
        # print "Initial story: ", story
        # print '\n'
        lemmatized_documents = self.lemmatize_story(story)
        topics = self.get_LDA_topics(lemmatized_documents)
        return topics
