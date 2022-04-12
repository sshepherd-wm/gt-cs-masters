""""""                   
"""                   
Template for implementing QLearner  (c) 2015 Tucker Balch                   
                   
Copyright 2018, Georgia Institute of Technology (Georgia Tech)                   
Atlanta, Georgia 30332                   
All Rights Reserved                   
                   
Template code for CS 4646/7646                   
                   
Georgia Tech asserts copyright ownership of this template and all derivative                   
works, including solutions to the projects assigned in this course. Students                   
and other users of this template code are advised not to share it with others                   
or to make it available on publicly viewable websites including repositories                   
such as github and gitlab.  This copyright statement should not be removed                   
or edited.                   
                   
We do grant permission to share solutions privately with non-students such                   
as potential employers. However, sharing with other current or future                   
students of CS 7646 is prohibited and subject to being investigated as a                   
GT honor code violation.                   
                   
-----do not edit anything above this line---                   
                   
Student Name: Tucker Balch (replace with your name)                   
GT User ID: tb34 (replace with your User ID)                   
GT ID: 900897987 (replace with your GT ID)                   
"""                   
                   
import random as rand                   
                   
import numpy as np                   
                   
                   
class QLearner(object):                   
    """                   
    This is a Q learner object.                   
                   
    :param num_states: The number of states to consider.                   
    :type num_states: int                   
    :param num_actions: The number of actions available..                   
    :type num_actions: int                   
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.                   
    :type alpha: float                   
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.                   
    :type gamma: float                   
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.                   
    :type rar: float                   
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.                   
    :type radr: float                   
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.                   
    :type dyna: int                   
    :param verbose: If “verbose” is True, your code can print out information for debugging.                   
    :type verbose: bool                   
    """                   
    def __init__(                   
        self,                   
        num_states=100,                   
        num_actions=4,                   
        alpha=0.2,                   
        gamma=0.9,                   
        rar=0.5,                   
        radr=0.99,                   
        dyna=0,                   
        verbose=False,                   
        ):                   
        """                   
        Constructor method                   
        """                   
        self.verbose = verbose
        self.num_states = num_states                   
        self.num_actions = num_actions                   
        self.s = 0 ## current state                  
        self.a = 0 ## current action
        self.actions = list(range(num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        ## tables for remembering actions and rewards
        # Q Table (look up next state and reward by state and action taken)
        self.qt = np.zeros((num_states,num_actions))

        if self.dyna > 0: # if Dyna simulation
            # Dyna transition occurence tracking
            self.dt = np.full((num_states,num_actions,num_states), .00001)
            # Dyna reward tracking
            self.dr = np.zeros((num_states,num_actions))

                   
    def querysetstate(self, s):                   
        """                   
        Update the state without updating the Q-table                   
                   
        :param s: The new state                   
        :type s: int                   
        :return: The selected action                   
        :rtype: int                   
        """
        ## set the state            
        self.s = s
        
        ## do random action
        #action = rand.randint(0, self.num_actions - 1)
        action = np.random.choice(self.actions)
        self.a = action
        
        ## return the action
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action
                   
    def query(self, s_prime, r):                   
        """                   
        Update the Q table and return an action                   
                   
        :param s_prime: The new state                   
        :type s_prime: int                   
        :param r: The immediate reward                   
        :type r: float                   
        :return: The selected action                   
        :rtype: int                   
        """
        
        ## Update Dyna simulated learning tables
        if self.dyna > 0:
            ## update transition occurrence table
            self.dt[self.s][self.a][s_prime] += 1
            ## update reward table
            self.dr[self.s][self.a] *= (1 - self.alpha)
            self.dr[self.s][self.a] += self.alpha * r

        ## get a_prime (best action based on state_prime)
        ## ACTION
        # choose random action according to rar, or choose an informed action
        a_prime = self._choose_best_action(s_prime)

        ## calculate total reward (combination of old estimate + immediate and future, adjusted by learning rate)
        reward =  (1 - self.alpha) * self.qt[self.s][self.a] # existing reward
        reward += self.alpha * (r + self.gamma * self.qt[s_prime][a_prime]) # updated reward value
        #reward =  self.qt[self.s][self.a][1] # existing reward
        #reward += self.alpha * (r + self.qt[s_prime][a_prime][1] - reward) # updated reward value

        ## update record effect of previous action with next state reward
        self.qt[self.s][self.a] = reward

        ## simulate with Dyna
        if self.dyna > 0:
            self.dyna_simulate()

        ## update random action rate based on decay
        self.rar = self.rar * self.radr

        ## get action (is this right after calculating a_prime???)
        if rand.random() <= self.rar:
            action = np.random.choice(self.actions)
        else:
            action = a_prime

        ## update current state and action
        self.s = s_prime
        self.a = action
              
        if self.verbose:                   
            print(f"s = {s_prime}, a = {action}, r={r}")                   
        return action

    def query_without_update(self, state):
        return self._choose_best_action(state) 

    def dyna_simulate(self):

        ## randomly initialize some state,action pairs
        state_simulate  = np.random.randint(low=0, high=self.num_states,  size=self.dyna)
        action_simulate = np.random.randint(low=0, high=self.num_actions, size=self.dyna)

        ## query dt for the randomly generaged state,action pairs
        tsim = self.dt[state_simulate,action_simulate]

        ## get the most frequent occurring resulting state for each s,a pair from dyna transition tracking
        s_primes = np.argmax(tsim, axis=1)

        ## get the rewards for the simulated state,action pairs from dyna reward tracking
        rewards = self.dr[state_simulate,action_simulate]
        rewards =  (
            (1 - self.alpha) * self.qt[state_simulate,action_simulate] # existing reward
            + self.alpha * (rewards + self.qt[s_primes,np.argmax( self.qt[s_primes], axis=1 )]) # updated reward value
        )

        ## update the Q table with the rewards
        self.qt[state_simulate,action_simulate] = rewards

    def _choose_best_action(self, s):
        '''
        Iterate through results for actions taken at the current state
        and choose the action that has the highest reward value
        (basically an argmax on the qt data structure for a given state)
        '''

        return np.argmax( self.qt[s], axis=0 )


    def author(self):
        return 'sshepherd35'        
                   
                   
if __name__ == "__main__":                   
    print("Remember Q from Star Trek? Well, this isn't him")                   
