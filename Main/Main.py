from Data.Data import Data
from Recommender.Recommender import RecSys

dt = Data()

games = dt.getGameData()
users = dt.getUserData()

recommender = RecSys(games, users)




# print(games)
# print(users)
