#####################################################################
# Author: Sandhya Saisubramanian
# Description: Implements value iteration
#####################################################################
import numpy as np
import sys
import timeit

class Env():
    def __init__(self, states, actions,P,R, start_state_id,goal_state_id):
        self.gamma = 1.0
        self.states = states
        self.actions =  actions
        self.P = P
        self.R = R
        self.goal_id = goal_state_id
        self.start_state_id = start_state_id

        self.V = np.zeros((len(self.states))).astype('float32').reshape(-1,1)
        self.Q =  np.zeros((len(self.states), len(self.actions))).astype('float32')
        self.policy = {}

    def solve(self, solver,max_trials=700):
        if solver == "VI":
            start_timer = timeit.default_timer()
            self.policy = self.VI(max_trials)
            end_timer = timeit.default_timer()
            # print('Time taken to solve (seconds): ', end_timer - start_timer)
            return self.policy, float(self.V[self.start_state_id])
        else:
            raise ValueError('Unknown solver')

    def VI(self,trials):
        epsilon = 0.001
        max_trials = trials
        dead_end_cost = 500
        dead_end_states = []

        curr_iter = 0
        bestAction = np.full((len(self.states)), -1)

        while curr_iter < max_trials:
            max_residual = 0
            curr_iter += 1
            for ns, s in enumerate(self.states):
                if ns == self.goal_id:
                    bestAction[ns] = 0
                    continue
                bestQ = dead_end_cost
                
                hasAction = False
                for na, a in enumerate(self.actions):
                    if self.P[ns][na] is  None:
                        continue
                    if s in dead_end_states:
                        break

                    hasAction = True
                    qaction = min(dead_end_cost, self.qvalue(ns, na)) # Cost-minimization
                    self.Q[ns][na] = qaction 

                    if qaction < bestQ:
                        bestQ = qaction
                        bestAction[ns] = na

                if bestQ >= dead_end_cost or hasAction == False:
                    if s not in dead_end_states:
                        dead_end_states.append(s)
                        self.V[ns] = dead_end_cost

                residual = abs(bestQ - self.V[ns])
                self.V[ns] = bestQ
                max_residual = max(max_residual, residual)

                

            if max_residual < epsilon:
                break

        return bestAction


    def qvalue(self,state_id, action_id):
        qaction = 0
        succ_list = self.P[state_id][action_id]
        if succ_list is not None:
            for succ in succ_list:
                succ_state_id = self.states.index(succ[0])
                prob = succ[1]
                qaction += prob * float(self.V[succ_state_id])
            return ((self.gamma * qaction) + self.R[state_id][action_id])

        else:
            return 500 

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

    def get_Trans_matrix(self):
        return self.T_matrix

    def get_R(self):
        return self.R

    def get_V(self):
        return self.V

    def get_Q(self):
        return self.Q

    def get_policy(self):
        return self.policy

    def get_gamma(self):
        return self.gamma

    def get_goal(self):
        return self.goal_id

    def get_P(self):
        return self.P


