import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json
import matplotlib.pyplot as plt
class RecSys:


    def __init__(self, games, users):
        self.games = games
        self.users = users
        self.top_rated = self.getTopRated(games, 10)
        print("Reading vectors...")
        self.users_vectors = pd.read_csv('C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Data\\user_vectors.csv', index_col=0)
        print("Done!")

        self.recommend(53875128)



    def recommend(self, user_id):
        # user_games = self.getReleventGames(user_id)

        rec_by_users = self.recommendByUsers(user_id, 5)


        # print(len(user_games))

    # Currently unused
    def getTagsClusters(self, games):
        vectors = self.createTfIDFVectors(games['tags'])
        clusters = self.applyKmeans(vectors)

        return clusters

    def createTfIDFVectors(self, tags):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(tags)
        return vectors.toarray()

    # Currently unused
    def applyKmeans(self, vectors):
        kMeans = KMeans(n_clusters=10, max_iter=10)

        print(vectors.shape)
        # Applying the K-Means algorithm
        print("Applying the K-Means algorithm")
        kMeans.fit(vectors)

        print(kMeans)

        result = pd.DataFrame(data=kMeans.labels_, columns=['cluster'])

        return result

    def getSimilarByTags(self, games, n, game_title):
        vectors = self.createTfIDFVectors(games['tags'])
        index = int(games.index[games['name'] == game_title][0])
        similarities = list(map(lambda x: cosine_similarity([x], [vectors[index]])[0][0], vectors))
        similarities = list(zip(range(len(similarities)), similarities))
        top = sorted(similarities, key=lambda x: x[1], reverse=True)[1:n+1]
        result = pd.DataFrame(list(map(lambda x: games.iloc[x[0]], top)))
        return result

    def getSimilar(self, games, n, game_title, column):
        value = games[games['name'] == game_title][column].tolist()[0]
        similarities = games[column].map(lambda x: abs(value - x)).tolist()
        similarities = list(zip(range(len(similarities)), similarities))
        top = sorted(similarities, key=lambda x: x[1], reverse=True)[1:n + 1]
        result = pd.DataFrame(list(map(lambda x: games.iloc[x[0]], top)))
        print(result)
        return result


    def getTopRated(self, games, n):
        return games.sort_values('ratings', ascending=False).head(n)

    def getReleventGames(self, user_id):
        user_games = self.users[self.users['user_id'] == user_id]
        user_games = user_games.sort_values('hours_played', ascending=False)

        us_titles = user_games['game_title'].tolist()
        gm_titles = self.games['name'].tolist()
        played = []
        for title in us_titles:
            if title in gm_titles:
                played.append(title)

        return played


    # This method
    def recommendByUsers(self, user_id, n):
        # Calculating n most similar users
        similar_users = self.getSimilarUsers(user_id, n)

        # Joining all games played by similar users
        joined = list(map(lambda x: self.users[self.users['user_id'] == x]['game_title'], similar_users))
        joined = pd.DataFrame(pd.concat(joined))

        # Creating a new table with game titles and the amount of duplicates
        joined = pd.DataFrame(joined.pivot_table(index=['game_title'], aggfunc='size'))
        duplicates = list(map(lambda x: x[0], joined.values))
        result = pd.DataFrame()
        result['game_title'] = pd.Series(joined.index)
        result['dups'] = pd.Series(duplicates)

        # Sorting games by number of duplicates so that the most frequent played are at the top
        result.sort_values('dups', ascending=False, inplace=True)

        # Returning top n results
        return result['game_title'].head(10)





    # This method uses user-vectors to calculate top n similar users
    # Input: user id and a number n
    # Output: returns top n users(list of id's) similar to given id
    def getSimilarUsers(self, user_id, n):
        # Creating vectors for given user and all other users as well
        vec = np.array(self.users_vectors.loc[user_id])
        vectors = self.users_vectors.to_numpy()
        # Calculating cosine similarity
        similarities = list(map(lambda x: cosine_similarity([x], [vec])[0][0], vectors))
        # Adding indexes before sorting
        similarities = list(zip(range(len(similarities)), similarities))
        # Sorting and slicing out top n
        top = sorted(similarities, key=lambda x: x[1], reverse=True)[1:n + 1]
        # Keeping indexes
        top = list(map(lambda x: x[0], top))
        # Converting to keys
        top = list(self.users_vectors.iloc[top].index)
        return top



