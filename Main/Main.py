from Data.Data import Data
from Recommender.Recommender import RecSys
from Evaluation.Evaluation import Evaluator
from Interface.Interface import GUI


# This method sets up recommender system and data
def loadSystem():
    print("Loading user and game information...")
    dt = Data()
    print("Done!")
    print("Setting up recommender system...")
    rec = RecSys(dt.games, dt.users, top_n=10, models={'collab': 0.55, 'tags': 0.35, 'price': 0.1})
    return dt.games, dt.users, rec

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


# Running application
# main()


# Running evaluation
dt = Data()
rec = RecSys(dt.games, dt.users, top_n=10,
             models={'collab': 0.4, 'tags': 0.5, 'price': 0.1})

eval = Evaluator(rec)
eval.evaluate()
eval.report()



