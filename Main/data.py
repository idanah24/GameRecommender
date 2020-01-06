import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing



# noinspection SpellCheckingInspection
def getUserData():
    user_info = pd.read_csv('steam-200k.csv')
    # Dropping rows with 'purchase' values
    user_info = user_info[user_info.play_or_purchase != 'purchase']
    # Dropping unnecessary columns
    user_info.drop(labels=['0', 'play_or_purchase'], axis='columns', inplace=True)

    # Normalizing 'hours played' column
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    user_info['hours_played'] = scaler.fit_transform(np.array(user_info['hours_played']).reshape(-1, 1))

    print(user_info)

    return user_info


# noinspection SpellCheckingInspection
def getGameData():
    game_info = pd.read_csv('steam.csv')

    #  Dropping unnecessary columns
    game_info.drop(
        labels=['release_date', 'english', 'developer', 'publisher',
                'required_age', 'achievements', 'owners', 'price', 'median_playtime', 'average_playtime',
                'appid', 'platforms'],
        axis='columns', inplace=True)

    # Calculating new ratings column
    game_info['ratings'] = game_info['positive_ratings'] - game_info['negative_ratings']

    # Dropping positive and negative ratings
    game_info.drop(labels=['positive_ratings', 'negative_ratings'], axis='columns', inplace=True)

    categories = list(game_info['categories'])
    genres = list(game_info['genres'])
    steamspy_tags = list(game_info['steamspy_tags'])

    tags = []

    # Joining categories, genres and steam tags into one tags column
    for i in range(len(categories)):
        tokens = list(set(categories[i].split(';') + genres[i].split(';') + steamspy_tags[i].split(';')))
        tags.append(tokens)

    # Creating the tags column
    game_info['tags'] = tags

    # Dropping columns that were joined
    game_info.drop(labels=['categories', 'genres', 'steamspy_tags'], axis='columns', inplace=True)

    # Normalize ratings column
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(1, 10))
    game_info['ratings'] = scaler.fit_transform(np.array(game_info['ratings']).reshape(-1, 1))

    return game_info




users = getUserData()
games = getGameData()

game_titles = set(list(games['name'].unique()) + list(users['game_title'].unique()))

# print(games.head(3))

# print(games.head(100).drop(labels='tags', axis=1))


