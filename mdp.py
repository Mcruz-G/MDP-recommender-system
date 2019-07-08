import operator
import pickle
import os
import statistics as st

from mdpHandler import MDPInitializer


class MDP:
    """
    Class to run the MDP.
    """

    def __init__(self, path='data', alpha=0.7, k=3, discountFactor=0.9, verbose=True, savePath="saved-models"):
        """
        The constructor for the MDP class.
        :param path: path to data
        :param alpha: the proportionality constant when considering transitions
        :param k: the number of items in each state
        :param discountFactor: the discount factor for the MDP
        :param verbose: flag to show steps
        :param savePath: the path to which models should be saved and loaded from
        """

        # Initialize the MDPInitializer
        self.mdp_i = MDPInitializer(path, k, alpha)
        self.df = discountFactor
        self.verbose = verbose
        self.savePath = savePath
        # The set of states
        self.S = {}
        # The set of state values
        self.V = {}
        # The set of actions
        self.A = []
        # The set of transitions
        self.T = {}

        self.policyVector = []
        # The policy of the MDP
        self.policy = {}
        # A policy list
        self.policyList = {}

    def printProgress(self, message):
        if self.verbose:
            print(message)

    def initializeMDP(self):
        """
        The method to initialise the MDP.
        :return: None
        """

        # Initialising the actions
        self.printProgress("Getting set of actions.")
        self.A = self.mdp_i.actions
        self.printProgress("Set of actions obtained.")

        # Initialising the states, state values, policy
        self.printProgress("Getting states, state-values, policy.")
        self.S, self.V, self.policy, self.policyList = self.mdp_i.generateInitialStates()
        self.printProgress("States, state-values, policy obtained.")

        # Initialise the transition table
        self.printProgress("Getting transition table.")
        self.T = self.mdp_i.generateTransitions(self.S, self.A)

        self.printProgress("Transition table obtained.")

    def oneStepLookAhead(self, state):
        """
        Helper function to calculate state-value function.
        :param state: state to consider
        :return: action values for that state
        """

        # Initialise the action values and set to 0
        actionValues = {}
        for action in self.A:
            actionValues[action] = 0

        # Calculate the action values for each action
        for action in self.A:
            for nextState, P_and_R in self.T[state][action].items():
                if nextState not in self.V:
                    self.V[nextState] = 0
                # action_value +=  probability * (reward + (discount * nextState_value))
                actionValues[action] += P_and_R[0] * (P_and_R[1] + (self.df * self.V[nextState]))
            
        return actionValues

    def updatePolicy(self):
        """
        Helper function to update the policy based on the value function.
        :return: None
        """

        for state in self.S:
            actionValues = self.oneStepLookAhead(state)

            # The action with the highest action value is chosen
            self.policy[state] = max(actionValues.items(), key=operator.itemgetter(1))[0]
            self.policyList[state] = sorted(actionValues.items(), key=lambda kv: kv[1], reverse=True)

    def policyEval(self):
        """
        Helper function to evaluate a policy
        :return: estimated value of each state following the policy and state-value
        """

        # Initialise the policy values
        policyValue = {}
        for state in self.policy:
            policyValue[state] = 0

        # Find the policy value for each state and its respective action dictated by the policy
        for state, action in self.policy.items():
            for nextState, P_and_R in self.T[state][action].items():
                if nextState not in self.V:
                    self.V[nextState] = 0
                # policyValue +=  probability * (reward + (discount * nextState_value))
                policyValue[state] += P_and_R[0] * (P_and_R[1] + (self.df * self.V[nextState]))

        return policyValue

    def comparePolicy(self, policyPrev):
        """
        Helper function to compare the given policy with the current policy
        :param policyPrev: the policy to compare with
        :return: a boolean indicating if the policies are different or not
        """

        for state in policyPrev:
            # If the policy does not match even once then return False
            if policyPrev[state] != self.policy[state]:
                return False
        return True

    def policyIteration(self, maxIteration=1000, start_where_left_off=False, to_save=True):

        avPolicy = []
        """
        Algorithm to solve the MDP
        :param maxIteration: maximum number of iterations to run.
        :param start_where_left_off: flag to load a previous model(set False if not and filename otherwise)
        :param to_save: flag to save the current model
        :return: None
        """
        # Load a previous model
        if start_where_left_off:
            self.load(start_where_left_off)

        # Start the policy iteration
        policyPrev = self.policy.copy()

        for i in range(maxIteration):
            self.printProgress("Iteration " + str(i) )

            # Evaluate given policy
            self.V = self.policyEval()

            meanV = st.mean(list(self.V.values()))
            avPolicy.append(meanV)

            # Improve policy
            self.updatePolicy()

            # If the policy not changed over 10 iterations it converged
            if i % 10 == 0:
                if self.comparePolicy(policyPrev):
                    self.printProgress("Policy converged at iteration " + str(i+1))
                    break
                policyPrev = self.policy.copy()

        
        

        # Save the model
        # if to_save:
        #     self.save("mdp-model_k=" + str(self.mdp_i.k) + ".pkl")

        return avPolicy

    def save(self, filename):
        """
        Method to save the trained model
        :param filename: the filename it should be saved as
        :return: None
        """

        self.printProgress("Saving model to " + filename)
        os.makedirs(self.savePath, exist_ok=True)
        with open(self.savePath + "/" + filename, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """
        Method to load a previous trained model
        :param filename: the filename from which the model should be extracted
        :return: None
        """

        self.printProgress("Loading model from " + filename)
        try:
            with open(self.savePath + "/" + filename, 'rb') as f:
                tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)
        except Exception as e:
            print(e)

    def savePolicy(self, filename):
        """
        Method to save the policy
        :param filename: the filename it should be saved as
        :return: None
        """

        self.printProgress("Saving model to " + filename)
        os.makedirs(self.savePath, exist_ok=True)
        with open(self.savePath + "/" + filename, 'wb') as f:
            pickle.dump(self.policyList, f, pickle.HIGHEST_PROTOCOL)

    def loadPolicy(self, filename):
        """
        Method to load a previous policy
        :param filename: the filename from which the model should be extracted
        :return: None
        """

        self.printProgress("Loading model from " + filename)
        try:
            with open(self.savePath + "/" + filename, 'rb') as f:
                self.policyList = pickle.load(f)
        except Exception as e:
            print(e)

    def recommend(self, userId):
        """
        Method to provide recommendation to the user
        :param userId: the userId of a given user
        :return: the game that is recommended
        """

        # self.printProgress("Recommending for " + str(userId))
        pre = []
        for i in range(self.mdp_i.k - 1):
            pre.append(None)
        games = pre + self.mdp_i.transactions[userId]

        # for g in games[self.mdp_i.k-1:]:
        #     print(self.mdp_i.games[g], self.mdp_i.game_price[g])

        userState = ()
        for i in range(len(games) - self.mdp_i.k, len(games)):
            userState = userState + (games[i],)
        # print(self.mdp_i.game_price[self.policy[userState]])
        # return self.mdp_i.games[self.policy[userState]]

        recList = []
        for gameDetails in self.policyList[userState]:
            recList.append((self.mdp_i.games[gameDetails[0]], gameDetails[1]))

        return recList

    def evaluateDecayScore(self, alpha=10):
        """
        Method to evaluate the given MDP using exponential decay score
        :param alpha: a parameter in exponential decay score
        :return: the average score
        """

        transactions = self.mdp_i.transactions.copy()

        userCount = 0
        totalScore = 0
        # Generating a testing for each test case
        for user in transactions:
            totalList = len(transactions[user])
            if totalList == 1:
                continue

            score = 0
            for i in range(1, totalList):
                self.mdp_i.transactions[user] = transactions[user][:i]

                recList = self.recommend(user)
                recList = [rec[0] for rec in recList]
                m = recList.index(self.mdp_i.games[transactions[user][i]]) + 1
                score += 2 ** ((1 - m) / (alpha - 1))

            score /= (totalList - 1)
            totalScore += 100 * score
            userCount += 1

        return totalScore / userCount

    def evaluateRecommendationScore(self, m=20):
        """
        Function to evaluate the given MDP using exponential decay score
        :param m: a parameter in recommendation score score
        :return: the average score
        """

        transactions = self.mdp_i.transactions.copy()

        userCount = 0
        totalScore = 0
        # Generating a testing for each test case
        for user in transactions:
            totalList = len(transactions[user])
            if totalList == 1:
                continue

            item_count = 0
            for i in range(1, totalList):
                self.mdp_i.transactions[user] = transactions[user][:i]

                recList = self.recommend(user)
                recList = [rec[0] for rec in recList]
                rank = recList.index(self.mdp_i.games[transactions[user][i]]) + 1
                if rank <= m:
                    item_count += 1

            score = item_count / (totalList - 1)
            totalScore += 100 * score
            userCount += 1

        return totalScore / userCount


# if __name__ == '__main__':
#     rs = MDP(path='data-mini')
#     rs.load('mdp-model_k=2.pkl')
#     print(rs.evaluate_recommendation_score())
#     # print(rs.mdp_i.transactions)
