from Data.Data import Data
from Recommender.Recommender import RecSys


dt = Data()

games = dt.getGameData()
users = dt.getUserData()

recommender = RecSys(games, users)

# print(games['price'])


# print(games)
# print(users)

# existing_games = set()
# titles_from_games = games['name'].tolist()
# titles_from_users = users['game_title'].tolist()
#
# count = 0
#
# for title in titles_from_users:
#     if title in titles_from_games:
#         count += 1
#         existing_games.add(title)
#
# print("There is information about {0} games out of {1} needed".format(count, len(titles_from_users)))
# print(users)
# print(existing_games)
# print(len(existing_games))

