import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Evaluator:

    # This class handles evaluation of recommender system model
    # class constructor takes in a recommender system object
    def __init__(self, system):
        self.system = system
        self.OUTPUT_PATH = 'C:\\Users\\Idan\\PycharmProjects\\GameRecommender\\Evaluation\\report.txt'
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    # This is the main method, calculating hit rate with Leave-One-Out-Cross-Validation
    def evaluate(self):

        print("Evaluating system...")
        num_of_games = len(list(self.system.users['game_title'].unique()))
        hits, misses, checked = 0, 0, 1
        for index, row in self.system.users.iterrows():
            print("Attempt {0}/{1}".format(checked, self.system.users.shape[0]))

            # Leaving current row out
            expected = row
            self.system.users.drop(index=index, inplace=True)

            # Making recommendations
            recommendations = self.system.recommend(expected['user_id'])


            # Checking for hit or miss
            if expected['game_title'] in recommendations:   # Hit
                hits += 1
                self.tp += 1
                self.fp += self.system.top_n - 1
                self.tn += num_of_games - self.system.top_n
                print("Hit!")

            else:   # Miss
                misses += 1
                self.fn += 1
                self.fp += self.system.top_n
                self.tn += num_of_games - self.system.top_n - 1
                print("Miss!")


            # Putting interaction back in data
            self.system.users = self.system.users.append(expected)
            self.system.users.sort_index(inplace=True)

            print("Hits: {0} | Misses: {1}".format(hits, misses))
            checked += 1

            print("{0:.1}% of evaluation done".format((checked / self.system.users.shape[0])*100))



        print("Evaluation complete!")

    # This method outputs scores calculated in evaluation process
    def report(self):
        # Calculating metrics scores
        hit_rate = self.tp / len(list(self.system.users['user_id'].unique())) * 100
        accuracy = (self.tp + self.tn) / (self.tp + self.fn + self.fp + self.tn) * 100
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        f1_score = (2 * precision * recall) / (precision + recall)

        # Outputting results to console
        print("Evaluation report:")
        print("System information:")
        print("Collaborative model: {0} | Tags model: {1} | Price model: {2}"
              .format(self.system.models['collab'] * 100,
                      self.system.models['tags'] * 100,
                      self.system.models['tags'] * 100))
        print("Top {0} recommendations".format(self.system.top_n))
        print("Hit rate: {0:.2f}%".format(hit_rate))
        print("Accuracy: {0:.2f}%".format(accuracy))
        print("Precision: {0:.3f}".format(precision))
        print("Recall: {0:.3f}".format(recall))
        print("F1-Score: {0:.3f}".format(f1_score))
        print("End of report")


        # Outputting results to file
        with open(self.OUTPUT_PATH, 'w') as file:
            file.write("Evaluation report:\n")
            file.write("System information:\n")
            file.write("Collaborative model: {0} | Tags model: {1} | Price model: {2}\n"
                  .format(self.system.models['collab'] * 100,
                          self.system.models['tags'] * 100,
                          self.system.models['tags'] * 100))
            file.write("Top {0} recommendations\n".format(self.system.top_n))
            file.write("Hit rate: {0:.2f}%\n".format(hit_rate))
            file.write("Accuracy: {0:.2f}%\n".format(accuracy))
            file.write("Precision: {0:.3f}\n".format(precision))
            file.write("Recall: {0:.3f}\n".format(recall))
            file.write("F1-Score: {0:.3f}\n".format(f1_score))
            file.write("End of report\n")
