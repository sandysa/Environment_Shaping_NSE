#####################################################################
# Author: Sandhya Saisubramanian
# Description: Driving domain setup
#####################################################################
import numpy as np
import sys
import random
from domain_helper import readmap,WallLoc_driving,PotholeLoc_driving,ParseGrid_driving,get_nse_penalty_dr
from env import Env

random.seed(1000)


class Driving():
    def __init__(self, map, start_loc, goal_loc):
        self.map = map
        self.actions = ['up_fast','down_fast','left_fast','right_fast','up_slow','down_slow','left_slow','right_slow']
        self.policy = {}
        self.deadend_cost = 500
        self.start_state = start_loc
        self.goal_state =  goal_loc
        self.populateParam()

    def populateParam(self):
        # print("In populateParam")
        self.grid = readmap(self.map)
        self.walls, self.potholes, self.reduced_speed_regions, self.grid_width, self.grid_height  = ParseGrid_driving(self.grid)
        self.states = self.generateStates()
        self.R = self.populateR()
        self.P = self.populate_P()
        if self.start_state in self.walls or self.goal_state in self.walls:
            sys.exit("Invalid start,goal locations; one or both is a wall location.")

    def solve(self):
        start_state_index = self.state_index(self.start_state)
        goal_state_index = self.state_index(self.goal_state)
        environment = Env(self.states, self.actions, self.P, self.R, start_state_index, goal_state_index)
        self.policy, expected_cost = environment.solve("VI",1000)
        # self.printPolicy()
        return self.Policy_verbose(),expected_cost

    def solve_simulate(self,trials):
        start_state_index = self.state_index(self.start_state)
        goal_state_index = self.state_index(self.goal_state)
        environment = Env(self.states, self.actions, self.P, self.R, start_state_index, goal_state_index)
        self.policy, expected_cost = environment.solve("VI",1000)
        policy_verbose, visitation_frequency = self.simulate_policy(trials)
        return policy_verbose, visitation_frequency,expected_cost

    def simulate_policy(self,trials):
        policy_verbose = {}
        visitation_frequency = np.zeros((len(self.states))).astype('float32').reshape(-1,1)
        for i in range(trials):
            state_visit = np.zeros((len(self.states))).astype('float32').reshape(-1,1)
            ns = self.state_index(self.start_state)
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

    def Policy_verbose(self):
        policy_verbose = {}
        for ns, state in enumerate(self.states):
            policy_verbose[ns] = self.actions[self.policy[ns]]
        return policy_verbose

    def solve_feedback(self,disapproved_actions):
        start_state_index = self.state_index(self.start_state)
        goal_state_index = self.state_index(self.goal_state)
        for ns in disapproved_actions:
            disable_actions_list = disapproved_actions[ns]
            for a in disable_actions_list:
                a_index = self.actions.index(a)
                self.P[ns][a_index] = None
        environment = Env(self.states, self.actions, self.P, self.R, start_state_index, goal_state_index)
        self.policy, expected_cost = environment.solve("VI",1000)
        if expected_cost >= self.deadend_cost:
            print("returning None", expected_cost)
            return None, expected_cost
        return self.Policy_verbose(),expected_cost

    def generateStates(self):
        states = []
        for r in range(len(self.grid)):
            for c in range(len(self.grid[0])):
                if (r,c) in self.walls:
                    continue
                states.append((r,c))
        # print("number of states: %s"%(len(states)))
        return states
           
    # Driving fast costs 1 and driving slow costs 2.
    def populateR(self):
        R =  np.ones((len(self.states), len(self.actions)))
        goal_state_index = self.state_index(self.goal_state)
        for action in ['up_slow','down_slow','left_slow','right_slow']:
            aid = self.actions.index(action)
            R[:,aid] = 2
        R[goal_state_index,:] = 0
        return R

    def getSucc(self,state,action):
        succ_prob = 0.85
        fail_prob = 0.15
        succ = []
        if state in self.walls:
            return None
        x = state[0]
        y = state[1]

        if state == self.goal_state:
            succ.append((state,1))  
            return succ
        # Driving fast is disabled in reduced speed regions
        if state in self.reduced_speed_regions:
            if "fast" in action:
                return None

        if "right" in action:
            new_pos = (x,y+1)
            slide_down = (x+1,y)
            if new_pos in self.walls or (y+1) > self.grid_width:
                return None
            succ.append((new_pos,succ_prob))
            if slide_down in self.walls or (x+1) > self.grid_height:
                succ.append((state,fail_prob))
            else:
                succ.append((slide_down,fail_prob))
            return succ

        elif "left" in action:
            new_pos = (x,y-1)
            slide_up = (x-1, y)
            if new_pos in self.walls or (y-1) < 0:
                return None
            succ.append((new_pos,succ_prob))
            if slide_up in self.walls or (x-1) < 0:
                succ.append((state,fail_prob))
            else:
                succ.append((slide_up,fail_prob))
            return succ

        elif "up" in action:
            new_pos = (x - 1,y)
            slide_right = (x, y + 1)
            if new_pos in self.walls or (x-1) < 0:
                return None
            succ.append((new_pos,succ_prob))
            if slide_right in self.walls or (y+1) > self.grid_width:
                succ.append((state,fail_prob))
            else:
                succ.append((slide_right,fail_prob))
            return succ

        elif "down" in action:
            new_pos = (x+1, y)
            slide_left = (x,y-1)
            if new_pos in self.walls or (x+1) > self.grid_height:
                return None
            succ.append((new_pos,succ_prob))
            if slide_left in self.walls or (y-1) < 0:
                succ.append((state,fail_prob))
            else:
                succ.append((slide_left,fail_prob))
            return succ


    def checkTransProb(self, state,action):
        succ_list = self.getSucc(state,action)
        total_prob = 0
        if succ_list != None:
            for succ in succ_list:
                total_prob += succ[1]
                if succ[0] not in self.states:
                    print("State not found----",state, action, succ[0])
                if succ[0] in self.walls:
                    print("state is a wall", succ[0])
            if total_prob != 1.0:
                print(state,action, succ_list)
                sys.exit("Transition prob does not sum to 1")


    def populate_P(self):
        P = [ [None]*len(self.actions) for i in range(len(self.states)) ]
        for s, state in enumerate(self.states):
            for a, action in enumerate(self.actions):
                P[s][a] = self.getSucc(state, action)
                self.checkTransProb(state,action)
        return P

    def calculate_final_cost_NSE(self, policy,trials=100):
        NSE_locations = PotholeLoc_driving(self.grid)
        cost = np.zeros((trials)).astype('float32').reshape(-1,1)
        nse_incurred = np.zeros((trials)).astype('float32').reshape(-1,1)

        for i in range(trials):
            ns = self.state_index(self.start_state)
            while ns!= self.state_index(self.goal_state):
                state = self.states[ns]
                # print(i,ns, state)
                best_action = policy[ns]
                aid = self.actions.index(best_action)
                cost[i] += self.R[ns][aid]
                
                nse_incurred[i] += get_nse_penalty_dr(state,best_action,NSE_locations)
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

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

    def state_index(self,state):
        return self.states.index(state)

    def printPolicy(self):
        for ns,s in enumerate(self.states):
            print(s,self.actions[self.policy[ns]])

    def start_goal(self):
        print(self.start_state, self.goal_state)

