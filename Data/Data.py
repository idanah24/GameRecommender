import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Data:

    # This class handles data processing

    def __init__(self):
        pass

    # noinspection SpellCheckingInspection
    def getUserData(self):
        user_info = pd.read_csv('C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Data\\steam-200k.csv')

        # Dropping rows with 'purchase' values
        user_info = user_info[user_info.play_or_purchase != 'purchase']
        # Dropping unnecessary columns
        user_info.drop(labels=['0', 'play_or_purchase'], axis='columns', inplace=True)

        # Normalizing 'hours played' column
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        user_info['hours_played'] = scaler.fit_transform(np.array(user_info['hours_played']).reshape(-1, 1))


        return user_info

    # noinspection SpellCheckingInspection
    def getGameData(self):
        game_info = pd.read_csv('C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Data\\steam.csv')

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

