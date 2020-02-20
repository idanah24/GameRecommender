import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Data:

    # This class handles data processing

    def __init__(self, load=True):
        # Paths to save/load data
        self.RAW_USER_DATA = 'C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Data\\steam-200k.csv'
        self.RAW_GAME_DATA = 'C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Data\\steam.csv'

        self.READY_USER_DATA = 'C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Data\\users.csv'
        self.READY_GAME_DATA = 'C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Data\\games.csv'

        # Loading existing ready data
        if load:
            self.users, self.games = self.load()
        # Processing raw data
        else:
            self.users = self.getUserData()
            self.games = self.getGameData()
            self.process()


    # noinspection SpellCheckingInspection
    def getUserData(self):
        user_info = pd.read_csv(self.RAW_USER_DATA)
        user_info.columns = ['user_id', 'game_title', 'play_or_purchase', 'hours_played', '0']

        # Dropping rows with 'purchase' values
        user_info = user_info[user_info.play_or_purchase != 'purchase']
        # Dropping unnecessary columns
        user_info.drop(labels=['0', 'play_or_purchase'], axis='columns', inplace=True)

        return user_info

    # noinspection SpellCheckingInspection
    def getGameData(self):
        game_info = pd.read_csv(self.RAW_GAME_DATA)


        #  Dropping unnecessary columns
        game_info.drop(
            labels=['release_date', 'english', 'developer', 'publisher',
                    'required_age', 'achievements', 'owners', 'average_playtime',
                    'median_playtime',
                    'appid', 'platforms'],
            axis='columns', inplace=True)

        # Calculating new ratings column
        game_info['ratings'] = game_info['positive_ratings'] - game_info['negative_ratings']

        # Dropping positive and negative ratings
        game_info.drop(labels=['positive_ratings', 'negative_ratings'], axis='columns', inplace=True)

        # Parsing tags columns
        s1 = game_info['categories'].map(lambda x: x.split(';'))
        s2 = game_info['genres'].map(lambda x: x.split(';'))
        s3 = game_info['steamspy_tags'].map(lambda x: x.split(';'))

        # Creating new tags column
        game_info['tags'] = s1.combine(s2.combine(s3, lambda x, y: list(set(x + y))), lambda x, y: list(set(x + y))) \
            .map(lambda x: ', '.join(x))

        # Dropping columns that were joined
        game_info.drop(labels=['categories', 'genres', 'steamspy_tags'], axis='columns', inplace=True)

        # Normalize ratings column
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(1, 10))
        game_info['ratings'] = scaler.fit_transform(np.array(game_info['ratings']).reshape(-1, 1))

        return game_info


    def process(self):

        # Removing rows from users for games with no information
        bool_series = self.users['game_title'].isin(self.games['name'].unique())
        self.users.drop(bool_series[bool_series == False].index, axis='rows', inplace=True)

        # Removing users who only played one game
        users_to_remove = self.users.groupby('user_id').size()
        users_to_remove = users_to_remove[users_to_remove == 1]
        bool_series = self.users['user_id'].isin(users_to_remove.index)
        self.users.drop(bool_series[bool_series == True].index, axis='rows', inplace=True)

    def save(self):
        self.games.to_csv(self.READY_GAME_DATA)
        self.users.to_csv(self.READY_USER_DATA)

    def load(self):
        games = pd.read_csv(self.READY_GAME_DATA)
        games.drop(labels='Unnamed: 0', axis='columns', inplace=True)
        users = pd.read_csv(self.READY_USER_DATA)
        users.drop(labels='Unnamed: 0', axis='columns', inplace=True)
        return users, games









