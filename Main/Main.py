from Data.Data import Data
from Recommender.Recommender import RecSys
import random

dt = Data()

print("Processing data...")
games = dt.getGameData()
users = dt.getUserData()
print("Done!")

rec = RecSys(games, users, top_n=10, models={'collab': 0.55, 'tags': 0.35, 'price': 0.1})


user_ids = users['user_id'].unique()
random.shuffle(user_ids)
id = random.choice(user_ids)


print("Recommending for: {0}".format(id))


result = rec.recommend(id)


print(result)
print(users[users['user_id'] == id]['game_title'].tolist())




