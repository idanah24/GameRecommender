import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings("ignore")

class RecSys:

    # This class is the actual recommendation engine

    def __init__(self, games, users, top_n, models, build=False):
        self.TAG_SIMS_PATH = 'C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Recommender\\tag_sims.csv'
        self.PRICE_SIMS_PATH = 'C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Recommender\\price_sims.csv'
        self.USER_SIMS_PATH = 'C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Recommender\\user_sims.csv'

        self.games = games
        self.users = users
        self.top_n = top_n
        self.models = models

        self.top_rated = self.getTopRated(games)
        self.games = self.getCommonGames(games)

        if build:
            self.buildModels()

        print("Reading tags...")
        self.tag_sims = pd.read_csv(self.TAG_SIMS_PATH).drop(labels=['Unnamed: 0'], axis='columns')
        print("Done!")
        print("Reading price...")
        self.price_sims = pd.read_csv(self.PRICE_SIMS_PATH).drop(labels=['Unnamed: 0'], axis='columns')
        print("Done!")
        print("Reading users...")
        self.user_sims = pd.read_csv(self.USER_SIMS_PATH)
        self.user_sims.index = self.user_sims['user_id']
        self.user_sims.drop(labels=['user_id'], axis='columns', inplace=True)
        print("Done!")

        # print(self.tag_sims)
        # print(self.price_sims)
        # print(self.user_sims)

        print("System Ready!")



    # This is the main method, recommending video games
    # Input: user id number
    # Output: top n (defined in class) recommendations
    def recommend(self, user_id):
        rec_by_tags, rec_by_price = None, None
        # Getting user games
        user_games = self.users[self.users['user_id'] == user_id].sort_values(by='hours_played',
                                                                              ascending=False)['game_title'].tolist()
        # Performing collaborative filtering
        rec_by_users = self.recommendByUsers(user_id, user_games)

        # Calculating score
        rec_by_users['score'] = rec_by_users['score'].map(lambda x: (self.models['collab'] * 100) / x)

        # If the games that the user played exist in database
        if user_games:
            # TODO: Consider picking users games randomly

            # Selecting the longest played game by user
            longest = user_games.pop(0)

            # Getting recommendations by tags
            rec_by_tags = self.getSimilar(game_title=longest, user_games=user_games, model='tags')

            # Adding longest game to the end of list
            user_games.append(longest)

            longest = user_games.pop(0)
            # Getting recommendations by price
            rec_by_price = self.getSimilar(game_title=longest, user_games=user_games, model='price')
            user_games.append(longest)


            # Calculating scores
            rec_by_tags['score'] = rec_by_tags['score'].map(lambda x: (self.models['tags'] * 100) / x)
            rec_by_price['score'] = rec_by_price['score'].map(lambda x: (self.models['price'] * 100) / x)

        # Joining all recommendations
        rec = rec_by_users.append(rec_by_tags.append(rec_by_price))

        # Giving duplicates higher ranking and selecting top n
        rec = list(rec.groupby(['name']).sum(). \
            sort_values(by='score', ascending=False).head(self.top_n).index)

        # Adding top rated games if there isn't enough recommendations
        if len(rec) < self.top_n:
            rec = rec + self.top_rated[0: self.top_n - len(rec)]

        return rec

    # This method creates tf-idf vectors
    # Input: a series of tags lists
    # Output: a matrix of tf-idf vectors
    def createTfIDFVectors(self, tags):

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(tags)
        return vectors.toarray()

    # This method calculates most similar games by a given model column
    # Input: game title to look for similar, user's already played games and the model
    # Output: top n most similar games to the given one
    def getSimilar(self, game_title, user_games, model):
        # Getting game index
        game_index = self.games[self.games['name'] == game_title].index[0]

        if model == 'tags':
            similar = self.tag_sims[str(game_index)]

        elif model == 'price':
            similar = self.price_sims[str(game_index)]

        # Sorting games similar to given game, dropping the game itself
        similar = similar.sort_values(ascending=False).drop(index=game_index)

        # Filtering out games user has already played and taking top n
        existing = self.games['name'].isin(user_games)
        similar = similar.drop(index=existing[existing == True].index).head(self.top_n)

        # Getting games names and preparing result
        result = self.games.iloc[similar.index].reset_index(drop=True)
        result['score'] = pd.Series(range(1, self.top_n + 1))
        result = result[['name', 'score']]

        return result


    # This method generates the most highly rated games in the dataset
    # Input: games data
    # Output: top n most rated games
    def getTopRated(self, games):
        # Sorting out top n most popular games
        top_rated = pd.DataFrame(games.sort_values('ratings', ascending=False).
                                 head(self.top_n)['name'])['name'].tolist()
        return top_rated



    # This method performs collaborative filtering
    # Input: user id number
    # Output: top n games played by most similar users
    def recommendByUsers(self, user_id, user_games):

        # Calculating n most similar users
        similar_users = self.getSimilarUsers(user_id)

        # Getting all games played by similar users that the user didn't play
        others_games = self.users['user_id'].isin(similar_users.index) & ~self.users['game_title'].isin(user_games)
        others_games = others_games[others_games == True].index
        others_games = self.users.iloc[others_games]

        # Adding score - ranking games played by very similar users higher
        others_games['score'] = others_games.apply(lambda row: similar_users.loc[row['user_id']], axis='columns')

        # Sorting and selecting top n
        others_games = others_games.groupby(['game_title']).sum().\
            sort_values(by='score', ascending=False)
        others_games['name'] = others_games.index
        result = others_games[['name', 'score']].reset_index(drop=True)
        result = result.head(self.top_n)

        # Adding new score => position in table
        result['score'] = pd.Series(range(1, self.top_n + 1))

        return result


    # This method uses user-vectors to calculate top n similar users
    # Input: user id and a number n
    # Output: returns top n users(list of id's) similar to given id
    def getSimilarUsers(self, user_id):
        top = self.user_sims[str(user_id)].sort_values(ascending=False)
        top.drop(labels=user_id, inplace=True)
        return top.head(self.top_n)

    def getCommonGames(self, games):
        relevant_games = self.games['name'].isin(self.users['game_title'].unique())
        relevant_games = self.games.iloc[relevant_games[relevant_games == True].index].reset_index(drop=True)
        return relevant_games

    def buildModels(self):

        print("Building models...")
        # Creating similarities by tags
        vectors = self.createTfIDFVectors(self.games['tags'])
        tag_sims = self.getSimilarities(vectors=vectors, method='cosine')

        # Creating similarities by price
        price_sims = self.getSimilarities(vectors=self.games['price'].to_numpy().reshape(-1, 1), method='euclidean')

        # Creating user similarities
        user_sims = pd.pivot_table(data=self.users, values='hours_played', index='user_id',
                                   columns='game_title', fill_value=0)
        index = user_sims.index
        user_sims = self.getSimilarities(user_sims.to_numpy(), method='cosine')
        user_sims.index, user_sims.columns = index, index

        print("Done!")

        # Saving data
        print("Saving tag similarities...")
        tag_sims.to_csv(self.TAG_SIMS_PATH)
        print("Done!")
        print("Saving price similarities...")
        price_sims.to_csv(self.PRICE_SIMS_PATH)
        print("Done!")
        print("Saving user similarities...")
        user_sims.to_csv(self.USER_SIMS_PATH)
        print("Done!")

    def getSimilarities(self, vectors, method):
        if method == 'cosine':
            sims = cosine_similarity(vectors)

        elif method == 'euclidean':
            sims = euclidean_distances(vectors)


        sims = pd.DataFrame(sims)
        return sims

