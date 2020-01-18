from Data.Data import Data
from Recommender.Recommender import RecSys
from Evaluation.Evaluation import Evaluator
from Interface.Interface import GUI




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


main()


