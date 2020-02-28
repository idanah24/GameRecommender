import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Evaluator:

    # This class handles evaluation of recommender system model
    # class constructor takes in a recommender system object
    def __init__(self, system):
        self.system = system

    # This is the main method, calculating hit rate with Leave-One-Out-Cross-Validation
    def evaluate(self):

        print("Evaluating system...")
        hits, misses, checked = 0, 0, 1
        for index, row in self.system.users.iterrows():
            print("Attempt {0}/{1}".format(checked, self.system.users.shape[0]))

            # Leaving current row out
            expected = row
            self.system.users.drop(index=index, inplace=True)

            # Making recommendations
            recommendations = self.system.recommend(expected['user_id'])


            # Checking for hit or miss
            if expected['game_title'] in recommendations:
                hits += 1
                print("Hit!")

            else:
                misses += 1
                print("Miss!")


            # Putting interaction back in data
            self.system.users = self.system.users.append(expected)
            self.system.users.sort_index(inplace=True)

            print("Hits: {0} | Misses: {1}".format(hits, misses))
            checked += 1



        print("Evaluation complete!")
