import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans



class RecSys:


    def __init__(self, games, users):
        self.games = games
        self.users = users
        self.tags_clusters = self.getTagsClusters(games)


    def getTagsClusters(self, games):
        vectors = self.createTfIDFVectors(games['tags'])
        clusters = self.applyKmeans(vectors)

        return clusters

    def createTfIDFVectors(self, tags):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(tags)
        return vectors.toarray()



    def applyKmeans(self, vectors):
        kMeans = KMeans(n_clusters=10, max_iter=10)

        print(vectors.shape)
        # Applying the K-Means algorithm
        print("Applying the K-Means algorithm")
        kMeans.fit(vectors)

        print(kMeans)

        result = pd.DataFrame(data=kMeans.labels_, columns=['cluster'])

        return result

