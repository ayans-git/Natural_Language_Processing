%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self):
        #Number of arms = 8 (image an octopus)
        #Arm values are normally distributed with Mean = 0, Variance = 1
        self.arm_values = np.random.normal(0,1,8) 
        self.K = np.zeros(8)
        self.est_values = np.zeros(8)
        
    def get_reward(self, action):
        #Noise term added to each reward
        #Noise is also normally distributed with Mean = 0, Variance = 1
        noise = np.random.normal(0,1)
        reward = self.arm_values[action] + noise
        return reward
    
    def choose_eps_greedy(self, epsilon):
        rand_num = np.random.random()
        if epsilon > rand_num:
            return np.random.randint(8)
        else:
            return np.argmax(self.est_values)
    
    #Continuously calculate average of rewards
    def update_est(self, action, reward):
        self.K[action] += 1
        alpha = 1./self.K[action]
        self.est_values[action] += alpha * (reward - self.est_values[action])
        
#An experiment to pull the arm Npulls times for a given 10 arm bandit
def experiment(bandit, Npulls, epsilon):
    history = []
    for i in range (Npulls):
        action = bandit.choose_eps_greedy(epsilon)
        R = bandit.get_reward(action)
        bandit.update_est(action, R)
        history.append(R)
    return history


#Repeat experiment multiple times and observe the evolution of the rewards, averaged over all the experiments
Nexp = 1000
Npulls = 6000

avg_outcome_eps0p0 = np.zeros(Npulls)
avg_outcome_eps0p01 = np.zeros(Npulls)
avg_outcome_eps0p1 = np.zeros(Npulls)

for i in range (Nexp):
    bandit = Bandit()
    avg_outcome_eps0p0 += experiment(bandit, Npulls, 0.0)
    bandit = Bandit()
    avg_outcome_eps0p01 += experiment(bandit, Npulls, 0.01)
    bandit = Bandit()
    avg_outcome_eps0p1 += experiment(bandit, Npulls, 0.1)
    
avg_outcome_eps0p0 /= np.float(Nexp)
avg_outcome_eps0p01 /= np.float(Nexp)
avg_outcome_eps0p1 /= np.float(Nexp)


plt.plot(avg_outcome_eps0p0, label = "eps = 0.0")
plt.plot(avg_outcome_eps0p01, label = "eps = 0.01")
plt.plot(avg_outcome_eps0p1, label = "eps = 0.1")
plt.ylim(0, 2.0)
plt.legend()
plt.show()
