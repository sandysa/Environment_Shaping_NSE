#####################################################################
# Author: Sandhya Saisubramanian
# Description: Boxpushing domain setup.
#####################################################################
import numpy as np
import sys
import random
from domain_helper import readmap,ParseGrid_boxpushing,get_nse_penalty_bp,CarpetLoc_boxpushing,VaseLocation
from env import Env

# random.seed(10000)

class Boxpushing():
    def __init__(self, map):
        self.map = map
        self.actions = ['up','down','left','right','pickup']
        self.grid = readmap(map)
        self.policy = {}
        self.deadend_cost = 500
        self.populateParam()
        

    def populateParam(self):
        self.s0, self.goal_loc, self.goal_state, self.box_init, self.wall = ParseGrid_boxpushing(self.grid)
        self.states = self.generateStates()
        self.R = self.populateR()
        self.P = self.populate_P()

    def solve(self):
        start_state_index = self.state_index(self.s0)
        goal_state_index = self.state_index(self.goal_state)
        environment = Env(self.states, self.actions, self.P, self.R, start_state_index, goal_state_index)
        self.policy, expected_cost = environment.solve("VI")
        return self.Policy_verbose(),expected_cost


    def solve_simulate(self,trials):
        start_state_index = self.state_index(self.s0)
        goal_state_index = self.state_index(self.goal_state)
        environment = Env(self.states, self.actions, self.P, self.R, start_state_index, goal_state_index)
        self.policy, expected_cost = environment.solve("VI")
        if expected_cost >= self.deadend_cost:
            return {}, np.zeros((len(self.states))).reshape(-1,1), expected_cost
        policy_verbose, visitation_frequency = self.simulate_policy(trials)

        return policy_verbose, visitation_frequency,expected_cost
        
    def solve_feedback(self,disapproved_actions):
        start_state_index = self.state_index(self.s0)
        goal_state_index = self.state_index(self.goal_state)
        for ns in disapproved_actions:
            disable_actions_list = disapproved_actions[ns]
            for a in disable_actions_list:
                a_index = self.actions.index(a)
                self.P[ns][a_index] = None
        environment = Env(self.states, self.actions, self.P, self.R, start_state_index, goal_state_index)
        self.policy, expected_cost = environment.solve("VI")
        if expected_cost >= self.deadend_cost:
            print("returning None", expected_cost)
            return None, expected_cost
        return self.Policy_verbose(),expected_cost


    def Policy_verbose(self):
        policy_verbose = {}
        for ns, state in enumerate(self.states):
            policy_verbose[ns] = self.actions[self.policy[ns]]
        return policy_verbose

    
    def simulate_policy(self,trials):
        random.seed(10000)
        policy_verbose = {}
        visitation_frequency = np.zeros((len(self.states))).astype('float32').reshape(-1,1)
        for i in range(trials):
            state_visit = np.zeros((len(self.states))).astype('float32').reshape(-1,1)
            ns = self.state_index(self.s0)
            while ns!= self.state_index(self.goal_state):
                state_visit[ns] = 1
                best_action = self.policy[ns]
                policy_verbose[ns] = self.actions[best_action]
                succ_list = self.P[ns][best_action]
                succ = random.choice(succ_list)
                ns = self.state_index(succ[0])
            policy_verbose[ns] = self.actions[self.policy[ns]]
            visitation_frequency += state_visit
        visitation_frequency = visitation_frequency/trials 
        return policy_verbose, visitation_frequency

    def generateStates(self):
        states = []
        for r in range(len(self.grid)):
            for c in range(len(self.grid[0])):
                states.append((r,c,True))
                states.append((r,c,False))
        # print("number of states: %s"%(len(states)))
        return states

    def state_index(self,state):
        return self.states.index(state)

    def populateR(self):
        R =  np.ones((len(self.states), len(self.actions)))
        goal_state_index = self.state_index(self.goal_state)
        R[goal_state_index,:] = 0
        return R

    def populate_P(self):
        P = [ [None]*len(self.actions) for i in range(len(self.states)) ]
        for s, state in enumerate(self.states):
            for a, action in enumerate(self.actions):
                P[s][a] = self.getSucc(state, action)
        return P

    def getSucc(self, state, action):
        succ_prob = 0.9
        fail_prob = 0.1
        succ = []
        if action == "pickup":
            if (state[0], state[1] ) == self.box_init and state[2] == False:
                sp = (state[0], state[1], True)
                succ.append((sp, 1.0))
                return succ
            else:
                return None

        # Moves up with success probability 0.9,
        # slides right with probability 0.1 or remains in the same state
        if action == "up":
            sp = (state[0] - 1, state[1], state[2]) #moves up
            sp_coord = (state[0] - 1, state[1])
            if sp in self.states and sp_coord not in self.wall:
                succ.append((sp, succ_prob))
                slide_right = (state[0], state[1] + 1, state[2])
                slide_right_coord = (state[0], state[1] + 1)
                if slide_right  in self.states and slide_right_coord not in self.wall:
                    succ.append((slide_right, fail_prob))
                    return succ
                else:
                    succ.append((state, fail_prob))
                    return succ
            else:
                return None

        # Moves down with success probability 0.9,
        # slides left with probability 0.1 or remains in the same state
        if action == "down":
            sp = (state[0] + 1, state[1], state[2]) #moves down
            sp_coord = (state[0] + 1, state[1])
            if sp in self.states and sp_coord not in self.wall:
                succ.append((sp, succ_prob))
                slide_left = (state[0], state[1] - 1, state[2])
                slide_left_coord = (state[0], state[1] - 1)
                if slide_left in self.states and slide_left_coord not in self.wall:
                    succ.append((slide_left, fail_prob))
                    return succ
                else:
                    succ.append((state, fail_prob))
                    return succ
            else:
                return None
        
        # Moves right with success probability 0.9,
        # slides down with probability 0.1 or remains in the same state
        if action == "right":
            sp = (state[0], state[1] + 1, state[2]) #moves right
            sp_coord = (state[0], state[1] + 1)
            if sp in self.states and sp_coord not in self.wall:
                succ.append((sp, succ_prob))
                slide_down = (state[0] + 1, state[1], state[2])
                slide_down_coord = (state[0] + 1, state[1])
                if slide_down in self.states and slide_down_coord not in self.wall:
                    succ.append((slide_down, fail_prob))
                    return succ
                else:
                    succ.append((state, fail_prob))
                    return succ
            else:
                return None

        # Moves left with success probability 0.9,
        # slides up with probability 0.1 or remains in the same state
        if action == "left":
            sp = (state[0], state[1] - 1, state[2]) #moves left
            sp_coord = (state[0], state[1] - 1)
            if sp in self.states and sp_coord not in self.wall:
                succ.append((sp, succ_prob))
                slide_up = (state[0] - 1, state[1], state[2])
                slide_up_coord = (state[0] - 1, state[1])
                if slide_up  in self.states and slide_up_coord not in self.wall:
                    succ.append((slide_up, fail_prob))
                    return succ
                else:
                    succ.append((state, fail_prob))
                    return succ
            else:
                return None
        return None
  
    
    def calculate_final_cost_NSE(self, policy,trials=100):
        random.seed(100)
        NSE_locations = CarpetLoc_boxpushing(self.grid)+VaseLocation(self.grid)
        cost = np.zeros((trials)).astype('float32').reshape(-1,1)
        nse_incurred = np.zeros((trials)).astype('float32').reshape(-1,1)

        for i in range(trials):
            ns = self.state_index(self.s0)
            while ns!= self.state_index(self.goal_state):
                state = self.states[ns]
                best_action = policy[ns]
                aid = self.actions.index(best_action)
                cost[i] += self.R[ns][aid]
                
                nse_incurred[i] += get_nse_penalty_bp(state,best_action,NSE_locations)
                succ_list = self.P[ns][aid]
                succ = random.choice(succ_list)
                ns = self.state_index(succ[0])

        avg_cost = np.sum(cost,axis=0)/trials
        avg_nse = np.sum(nse_incurred,axis=0)/trials
        std_cost = np.std(cost,axis=0)
        std_nse = np.std(nse_incurred,axis=0)

        return float(avg_cost), float(std_cost), float(avg_nse), float(std_nse)


    def generate_state_actions(self):
        sa_list = []
        for s,state in enumerate(self.states):
            for a, action in enumerate(self.actions):
                if self.P[s][a] != None:
                    temp = []
                    temp.append(s)
                    temp.append(state)
                    temp.append(action)
                    sa_list.append(temp)
        return sa_list

    def printPolicy(self):
        for ns,s in enumerate(self.states):
            print(s,self.actions[self.policy[ns]])

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions


