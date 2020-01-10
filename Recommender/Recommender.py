import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans



class RecSys:


    def __init__(self, type):
        pass



def createTfIDFVectors(games):

    vectorizer = TfidfVectorizer()
    vectors = pd.DataFrame(vectorizer.fit_transform(games['tags']).toarray())
    vectors['game_title'] = games['name']
    return vectors



def applyKmeans(vectors):
    kMeans = KMeans(n_clusters=10)
    vecs = vectors.drop(labels='game_title', axis='columns')
    vecs = np.array(vecs)

    kMeans.fit(vecs)

    result = pd.DataFrame(data=kMeans.labels_, columns=['cluster'])
    result['game_title'] = vectors['game_title']

    print(result[result.cluster == 1])

    return None

