from Data.Data import Data
from Recommender.Recommender import RecSys
from Evaluation.Evaluation import Evaluator
from Interface.Interface import GUI
import pandas as pd



def loadSystem():
    dt = Data()
    print("Processing data...")
    games = dt.getGameData()
    users = dt.getUserData()
    print("Done!")
    rec = RecSys(games, users, top_n=10, models={'collab': 0.55, 'tags': 0.35, 'price': 0.1})
    return games, users, rec





def main():
    games, users, rec = loadSystem()
    user_list = users['user_id'].unique().tolist()
    gui = GUI()

    def Recommend(user_input):
        user_id = int(user_input)
        print("Recommending for {0}".format(user_id))
        recommendations = rec.recommend(user_id)
        print("Done!")
        gui.resultWindow(recommendations)

    gui.mainWindow(choice_list=user_list, menu_button_action=Recommend)


# main()


dt = Data()
print("Processing data...")
games = dt.getGameData()
users = dt.getUserData()
print("Done!")
rec = RecSys(games, users, top_n=10, models={'collab': 0.55, 'tags': 0.35, 'price': 0.1})
#
# print("Evaluating...")
# eval = Evaluator(rec)
# eval.evaluate()
# print("Done!")

# to_drop = []
# for user in users['user_id'].unique().tolist():
#     amount = rec.getReleventGames(user)
#
#     if len(amount) <= 1:
#         to_drop.append(user)
#
# for del_user in to_drop:
#     print("Dropping user {0}".format(del_user))
#     rows_to_drop = users[users['user_id'] == del_user].index
#     for drop_index in rows_to_drop:
#         users.drop(drop_index, inplace=True)
#
# users.to_csv('new_users.csv')
#




new_users = pd.read_csv('new_users.csv')

# avg = new_users['hours_played'].mean()


sum = 0
for user in new_users['user_id'].unique().tolist():
    sum += len(rec.getReleventGames(user))

print("avg is = {0}".format(sum / len(new_users['user_id'].unique().tolist())))




