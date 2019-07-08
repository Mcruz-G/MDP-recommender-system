import csv
import random


class MDPInitializer:
    """
    Class to generate state space.
    """

    def __init__(self, dataPath, k, alpha):
        """
        The constructor for the MDPInitializer class.
        Parameters:
        :param dataPath: path to data
        :param k: the number of items in each state
        :param alpha: the proportionality constant when considering transitions
        """

        self.userPath = dataPath + "/users.csv"
        self.transactionPath = dataPath + "/transactions.csv"
        self.gamesPath = dataPath + "/games.csv"
        self.k = k
        self.alpha = alpha
        self.totalSequences = {}

        self.gameData = {}
        self.transactions = {}
        # Get user data and initialise transactions under each user
        self.fillUserData()
        # Store transactions as { user_id : { game_title : [ play, purchase, play, ... ], ... }, ... }
        self.fillTransactionData()

        self.actions, self.games, self.gamePrice = self.getActionData()
        self.numOfActions = len(self.actions)

    def fillUserData(self):
        """
        The method to fill user data.
        :return: None
        """

        with open(self.userPath) as auxUserDoc:
            userCsv= csv.reader(auxUserDoc)
            next(userCsv)
            for userRow in userCsv:
                self.transactions[userRow[0]] = []

    def fillTransactionData(self):
        """
        The method to fill the transactions for each user.
        :return: None
        """

        with open(self.transactionPath) as auxTransDoc:
            transactionCsv = csv.reader(auxTransDoc)
            next(transactionCsv)
            for transRow in transactionCsv:
                gameTitle = transRow[1]
                userId = transRow[0]
                value = transRow[3]
                if gameTitle not in self.transactions[userId]:
                    self.transactions[userId].append(gameTitle)
                if gameTitle not in self.gameData:
                    self.gameData[gameTitle] = [0, 0]
                self.gameData[gameTitle][0] += float(value)
                self.gameData[gameTitle][1] += 1

        for game in self.gameData:
            self.gameData[game] = self.gameData[game][0] / self.gameData[game][1]

    def getActionData(self):
        """
        The method to obtain all games which will be actions.
        :return: list of the games/actions
        """

        actions = []
        games = {}
        gamePrice = {}

        with open(self.gamesPath) as auxGamesDoc:
            gamesCsv = csv.reader(auxGamesDoc)
            next(gamesCsv)
            for row in gamesCsv:
                actions.append(row[0])
                games[row[0]] = row[1]
                gamePrice[row[0]] = int(row[2])
        return actions, games, gamePrice

    def generateInitialStates(self):
            """
            The method to generate an initial state space.
            :return: states and the corresponding value vector
            """

            states = {}
            stateValue = {}
            policy = {}
            policyList = {}

            for user in self.transactions:
                # Prepend Nones for first transactions
                pre = []
                for i in range(self.k - 1):
                    pre.append(None)
                games = pre + self.transactions[user]

                # Generate states of k items
                for i in range(0, len(games) - self.k + 1):
                    tempTup = ()
                    for j in range(self.k):
                        tempTup = tempTup + (games[i + j],)

                    if tempTup in states:
                        states[tempTup] = states[tempTup] + 1
                    else:
                        states[tempTup] = 1
                        stateValue[tempTup] = 0
                        policy[tempTup] = random.choice(self.actions)
                        policyList[tempTup] = random.sample(self.actions, len(self.actions))
                        for ind in range(len(policyList[tempTup])):
                            policyList[tempTup][ind] = (policyList[tempTup][ind], 1)

                # Generate states of k+1 items
                for i in range(0, len(games) - self.k - 1):
                    tempTup = ()
                    for j in range(self.k + 1):
                        tempTup = tempTup + (games[i + j],)
                    if tempTup in self.totalSequences:
                        self.totalSequences[tempTup] = self.totalSequences[tempTup] + 1
                    else:
                        self.totalSequences[tempTup] = 1

            return states, stateValue, policy, policyList

    def generateTransitions(self, states, actions):
            """
            The method to generate the transition table.
            :param states: the initial states
            :param actions: the actions/items that can be chosen
            :return: a dictionary with transition probabilities
            """

            # Initialize the transitions dict
            transitions = {}

            # rewardVec = []
            # Store transitions as { state: { action/item chosen: { next_state: (alpha * count, reward), ... }, ... }, ... }
            for state, stateCount in states.items():
                for action in actions:
                    # Compute the new state
                    newState = ()
                    for i in range(1, self.k):
                        newState = newState + (state[i],)
                    newState = newState + (action,)

                    # Compute the complete sequence
                    totalSequence = state + (action,)
                    # Find number of times the total sequence occurs
                    if totalSequence not in self.totalSequences:
                        totalSequenceCount = 1
                        self.totalSequences[totalSequence] = totalSequenceCount
                    else:
                        totalSequenceCount = self.totalSequences[totalSequence]

                    # Fill the transition probabilities
                    if state not in transitions:
                        transitions[state] = {}
                    if action not in transitions[state]:
                        transitions[state][action] = {}
                    # Need to alpha * transition[state][action][n_state] as the action corresponds to the desired state
                    transitions[state][action][newState] = (self.alpha * totalSequenceCount / stateCount,
                                                            self.reward(newState))

            # Adding the other possibilities and their probabilities for a particular action
            for state in transitions:
                for action in transitions[state]:
                    for a in actions:
                        # Compute the new state
                        newState = ()
                        for i in range(1, self.k):
                            newState = newState + (state[i],)
                        newState = newState + (a,)

                        # Need to beta * transition[state][a][n_state] as the action doesn't correspond to the desired state
                        if newState not in transitions[state][action]:
                            transitions[state][action][newState] = (self.beta(action, newState)
                                                                    * transitions[state][a][newState][0],
                                                                    self.reward(newState))
                        
                        # rewardVec.append(self.reward(newState))
                            

            # Normalizing the probabilities
            for state in transitions:
                for action in transitions[state]:
                    total = 0
                    for newState in transitions[state][action]:
                        total += transitions[state][action][newState][0]
                    for newState in transitions[state][action]:
                        oldTup = transitions[state][action][newState]
                        transitions[state][action][newState] = (oldTup[0] / total, oldTup[1])

                        # rewardVec.append(self.reward(newState))

            return transitions

    def beta(self, action, newState):
            """
            Method to calculate the beta required
            :param action: the action taken
            :param newState: the new state
            :return: beta
            """

            # The difference in number of hours per unit currency
            diff = abs((self.gameData[action] / self.gamePrice[action]) -
                    (self.gameData[newState[self.k - 1]] / self.gamePrice[newState[self.k - 1]]))
            return diff / 120

    def reward(self, state):
            """
            Method to calculate the reward for each state
            :param state: the state
            :return: the reward for the given state
            """

            spent = 0
            for i in range(len(state) - 1):
                if state[i] is None:
                    spent += 0
                else:
                    spent += self.gamePrice[state[i]]
            # The average amount spent before this purchase
            if not len(state) == 1:
                spent /= (len(state) - 1)
            y = spent / self.gamePrice[state[self.k - 1]]
            
            if y > 1:
                y = 1/y
            
            return (1 - y) * (self.gameData[state[self.k - 1]]) + y * (self.gamePrice[state[self.k - 1]])