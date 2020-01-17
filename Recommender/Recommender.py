import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

class RecSys:

    # This class is the actual recommendation engine

    def __init__(self, games, users, top_n, models):
        self.games = games
        self.users = users
        self.top_n = top_n
        self.models = models
        print("Initializing top rated...")
        self.top_rated = self.getTopRated(games)
        print("Done!")
        print("Reading vectors...")
        self.users_vectors = pd.read_csv('C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Data\\user_vectors.csv', index_col=0)
        print("Done!")
        print("System ready!")

    # This is the main method, recommending video games
    # Input: user id number
    # Output: top n (defined in class) recommendations
    def recommend(self, user_id):
        rec_by_tags, rec_by_price = None, None
        # Getting user games that exist in game data
        user_games = self.getReleventGames(user_id)
        # Performing collaborative filtering
        rec_by_users = self.recommendByUsers(user_id)

        # If the games that the user played exist in database
        if user_games:
            # Getting recommendations by tags from a given random game
            random.shuffle(user_games)
            rec_by_tags = self.getSimilarByTags(self.games, random.choice(user_games))
            # Getting recommendations by price from a given random game
            random.shuffle(user_games)
            rec_by_price = self.getSimilar(self.games, random.choice(user_games), 'price')
            # Calculating scores
            rec_by_tags['score'] = self.models['tags'] * 100
            rec_by_tags['score'] = rec_by_tags['score'] / pd.Series([1] + list(rec_by_tags.index))
            rec_by_price['score'] = self.models['price'] * 100
            rec_by_price['score'] = rec_by_price['score'] / pd.Series([1] + list(rec_by_price.index))

        # Calculating scores
        rec_by_users['score'] = self.models['collab'] * 100

        # Setting score according to place in table
        rec_by_users['score'] = rec_by_users['score'] / pd.Series([1] + list(rec_by_users.index))

        rec = rec_by_users
        # Combining all recommendations to one table
        if user_games:
            rec = rec.append(rec_by_tags.append(rec_by_price, ignore_index=True), ignore_index=True)

        # Filtering out games that are already played
        played = self.users[self.users['user_id'] == user_id]['game_title'].tolist()
        rec = rec[~rec['game_title'].isin(played)]

        # Summing scores of games that are at more than one table
        rec = rec.groupby(['game_title']).sum()

        # Sorting to get top results
        rec = rec.sort_values('score', ascending=False)

        # If there are not enough recommendations, appending most popular games
        if rec.shape[0] < self.top_n:
            rec.append(self.top_rated, ignore_index=True)

        return list(rec.head(self.top_n).index)

    # This method creates tf-idf vectors
    # Input: a series of tags lists
    # Output: a matrix of tf-idf vectors
    def createTfIDFVectors(self, tags):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(tags)
        return vectors.toarray()

    # This method calculates most similar games by tags column
    # Input: games data, and a single game title
    # Output: top n most similar games to the given one
    def getSimilarByTags(self, games, game_title):
        # Getting tf idf vectors
        vectors = self.createTfIDFVectors(games['tags'])
        # Getting given game title's index
        index = int(games.index[games['name'] == game_title][0])
        # Calculating cosine similarity and adding index before sorting
        similarities = list(map(lambda x: cosine_similarity([x], [vectors[index]])[0][0], vectors))
        similarities = list(zip(range(len(similarities)), similarities))
        # Sorting and slicing out top n
        top = sorted(similarities, key=lambda x: x[1], reverse=True)[1:self.top_n+1]
        # Creating list of top n game titles
        result = pd.DataFrame(list(map(lambda x: games.iloc[x[0]], top)))
        # Making uniform data
        result = result['name'].reset_index(drop=True).rename('game_title')
        result.index = range(1, self.top_n + 1)
        return pd.DataFrame(result)

    # This method calculates most similar games by a given numeric column
    # Input: games data, a single game_title
    # Output: a frame with most similar games
    def getSimilar(self, games, game_title, column):
        # Getting given game value in given column
        value = games[games['name'] == game_title][column].tolist()[0]
        # Calculating euclidean distance between given game and all the rest
        similarities = games[column].map(lambda x: abs(value - x)).tolist()
        # Adding index
        similarities = list(zip(range(len(similarities)), similarities))
        # Sorting and slicing top n games
        top = sorted(similarities, key=lambda x: x[1], reverse=True)[1:self.top_n + 1]
        # Creating frame with top n game titles
        result = pd.DataFrame(list(map(lambda x: games.iloc[x[0]], top)))
        # Making uniform data
        result = result['name'].rename('game_title')
        result.index = range(1, self.top_n + 1)
        return pd.DataFrame(result)

    # This method generates the most highly rated games in the dataset
    # Input: games data
    # Output: top n most rated games
    def getTopRated(self, games):
        # Sorting out top n most popular games
        top_rated = pd.DataFrame(games.sort_values('ratings', ascending=False).head(self.top_n)['name'])
        # Making the data uniform
        top_rated.index = range(1, self.top_n + 1)
        top_rated.rename(columns={'name': 'game_title'}, inplace=True)
        top_rated['score'] = range(1, self.top_n + 1)
        return top_rated

    # This method searches for games in game data from a user played games
    # Input: user id number
    # Output: a list of game titles which exists in game data
    def getReleventGames(self, user_id):
        # Getting user games and sorting by hours played
        user_games = self.users[self.users['user_id'] == user_id]
        user_games = user_games.sort_values('hours_played', ascending=False)
        # Creating lists from games in game data and from user's played games
        us_titles = user_games['game_title'].tolist()
        gm_titles = self.games['name'].tolist()
        # Calculating intersection of the lists
        # TODO: get rid of for loop
        played = []
        for title in us_titles:
            if title in gm_titles:
                played.append(title)
        return played

    # This method performs collaborative filtering
    # Input: user id number
    # Output: top n games played by most similar users
    def recommendByUsers(self, user_id):
        # Calculating n most similar users
        similar_users = self.getSimilarUsers(user_id)
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
        # Resetting index
        result.index = range(1, result.shape[0] + 1)
        result = pd.DataFrame(result['game_title'])
        # Returning top n results if there is at least n games, otherwise returning all results
        if result.shape[0] >= self.top_n:
            return result.head(self.top_n)
        return result

    # This method uses user-vectors to calculate top n similar users
    # Input: user id and a number n
    # Output: returns top n users(list of id's) similar to given id
    def getSimilarUsers(self, user_id):
        # Creating vectors for given user and all other users as well
        vec = np.array(self.users_vectors.loc[user_id])
        vectors = self.users_vectors.to_numpy()
        # Calculating cosine similarity
        similarities = list(map(lambda x: cosine_similarity([x], [vec])[0][0], vectors))
        # Adding indexes before sorting
        similarities = list(zip(range(len(similarities)), similarities))
        # Sorting and slicing out top n
        top = sorted(similarities, key=lambda x: x[1], reverse=True)[1:self.top_n + 1]
        # Keeping indexes
        top = list(map(lambda x: x[0], top))
        # Converting to keys
        top = list(self.users_vectors.iloc[top].index)
        return top










   # # Currently unused
   #  def applyKmeans(self, vectors):
   #      kMeans = KMeans(n_clusters=10, max_iter=10)
   #
   #      print(vectors.shape)
   #      # Applying the K-Means algorithm
   #      print("Applying the K-Means algorithm")
   #      kMeans.fit(vectors)
   #
   #      print(kMeans)
   #
   #      result = pd.DataFrame(data=kMeans.labels_, columns=['cluster'])
   #
   #      return result


# # Currently unused
# def getTagsClusters(self, games):
#     vectors = self.createTfIDFVectors(games['tags'])
#     clusters = self.applyKmeans(vectors)
#
#     return clusters


