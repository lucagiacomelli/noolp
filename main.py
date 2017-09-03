from doc_similarity import DocSimilarity

print('\nHello in Document Similarity and Topic Modelling!!\n')

doc1 = 'This is a function to test document_path_similarity.'
doc2 = 'Use this function to see if your code in doc_to_synsets and similarity_score is correct!'

print('document1: \"', doc1 , '\"') 
print('document2: \"', doc2 , '\"') 
doc_sim = DocSimilarity(doc1, doc2)

print(doc_sim.document_path_similarity())



