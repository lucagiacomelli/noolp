import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer
import re

class TopicModelling:

    def __init__(self, documents):
        self.documents = documents

    def LDA_model():
        vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
        # Fit and transform
        X = vect.fit_transform(self.documents)

        print(X.shape)
        ## We have 2000 documents and 901 terms (tokens) that appear in at least 20
        ## documents and have at least 3 letters
        #print(X)

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
        #id_map = dict((v, k) for k, v in vect.vocabulary_.items())
        #ldamodel.update(corpus2)
        list_of_list = list(ldamodel2.get_document_topics(corpus2))
        
        result = list_of_list[0]
        return result
