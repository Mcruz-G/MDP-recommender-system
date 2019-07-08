from mdp import MDP,MDPInitializer
from pytictoc import TicToc
import matplotlib.pyplot as plt 

t = TicToc() #create instance
t.tic() #Start timer

def main(userNum = '59945701'):

    agent = MDP(path = 'data-mini', k = 3) #Create instance for MDP class
    agent.initializeMDP()                  #Initialize States, Actions, Probabilites and initial Rewards
    rewardEvolution = agent.policyIteration()                # The algorithm that solves MDP 
    recommendation = agent.recommend(userNum)  #Use recommendation function 
    evaluationRS = agent.evaluateRecommendationScore() #Evaluation score
    evaluationED = agent.evaluateDecayScore()           #Another evaluation score
    return recommendation, evaluationRS, evaluationED, userNum, rewardEvolution

rec, evaRS, evaED, userName, rewardEvolution = main()





print(f'Recomendation for user {userName}: Buy {rec[0][0]}; It has immediate reward of: {rec[0][1]}.')

print(f'Model evaluation (Microsoft recomendation score, 2002) = {evaRS}')
print(f'Model evaluation (Exponential decay) = {evaED}')


t.toc() #Time elapsed since t.tic()

print(rewardEvolution)
plt.plot(rewardEvolution,'r-')
plt.title('Reward evolution vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Mean discounted reward')
plt.show()