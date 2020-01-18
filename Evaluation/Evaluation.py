import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Evaluator:

    def __init__(self, system):
        self.system = system
        self.games = system.games
        self.users = system.users
        self.user_list = system.users['user_id'].unique()
        self.hits = 0
        self.failed = 0


    def evaluate(self):

        count_users = 1
        for user in self.user_list:
            # Getting all the games the user played
            user_games = self.users[self.users['user_id'] == user]['game_title'].tolist()

            # Randomly picking a game to drop
            random.shuffle(user_games)
            dropped = random.choice(user_games)

            # Dropping a game
            dropped_index = self.users.loc[self.users['game_title'] == dropped].loc[self.users['user_id'] == user].index[0]
            self.users.drop(dropped_index, inplace=True)

            print("Recommending for: {0} out of {1}".format(count_users, len(self.user_list)))
            count_users += 1

            # Creating recommendations for current user
            recommendations = self.system.recommend(user)

            # Checking if dropped game was recommended or not
            if dropped in recommendations:
                self.hits += 1
            else:
                self.failed += 1

            print("Hits: {0} | Miss: {1}".format(self.hits, self.failed))



