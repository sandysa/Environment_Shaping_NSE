########################################################################################
# Author: Sandhya Saisubramanian
# Description: Identifies the best modification for an environment, given agent policy
########################################################################################
import numpy as np
from domain_helper import *
import time 

class Designer():
    def __init__(self, maps, modifications, costs, E0, delta, domain, states):
        self.environment_configurations = maps
        self.modifications = modifications
        self.costs_modification = costs
        self.E0 = E0
        self.domain = domain
        self.states = states
        self.current_map = E0
        self.delta = delta
        self.mild_NSE_penalty = 2
        self.severe_NSE_penalty = 5
        self.NSE_locations = {}

    def set_designer_delta(self,val):
        self.delta = val

    def populateParameters(self,policy,visitation_freq):
        self.visitation_freq = visitation_freq
        self.actor_policy = policy
        self.grid = readmap(self.current_map)
        if self.domain =="boxpushing":
            self.NSE_locations = CarpetLoc_boxpushing(self.grid)+VaseLocation(self.grid)
        elif self.domain =="driving":
            self.NSE_locations = PotholeLoc_driving(self.grid)
        self.current_NSE = self.getNSE(self.current_map,self.actor_policy)
        self.initial_NSE = self.current_NSE

    def getNSE(self,map, policy):
        nse = 0
        grid = readmap(map)
        if self.domain == "boxpushing":
            NSE_locations = CarpetLoc_boxpushing(grid)+VaseLocation(grid)
            for s, state in enumerate(self.states):
                if s not in policy:
                    continue
                action = policy[s]
                if is_NSE_boxpushing(state,action, NSE_locations):
                    nse += self.severe_NSE_penalty * self.visitation_freq[s]

        elif self.domain == "driving":
            NSE_locations = PotholeLoc_driving(grid)
            for s, state in enumerate(self.states):
                if s not in policy:
                    continue
                action = policy[s]
                if mild_NSE_driving(state,action, NSE_locations):
                    nse += self.mild_NSE_penalty * self.visitation_freq[s]
                elif severe_NSE_driving(state, action,NSE_locations):
                    nse += self.severe_NSE_penalty * self.visitation_freq[s]
        return nse


    def best_design(self, visited, policy):
        best_util = 0
        best_modification = 'null'
        best_NSE = self.initial_NSE
        best_map =  self.current_map
        for modification in self.modifications:
            if modification not in visited:
                utility, updated_NSE = self.utility(modification, policy)
                if utility > best_util:
                    best_modification = modification
                    best_util = utility
                    best_NSE = updated_NSE
                elif utility == best_util and updated_NSE < best_NSE:
                    best_modification = modification
                    best_util = utility
                    best_NSE = updated_NSE
        print("best modification = %s, utility= %s, NSE = %s"%(best_modification, best_util, best_NSE))
        return best_util, best_modification, best_NSE

    def best_design_multiple_actors(self,visited, policies):
        best_util = 0
        initial_total_NSE = 0
        best_map = self.current_map
        best_modification = 'null'
        NSE_actors = []

        for p in policies:
            nse_value = self.getNSE(self.current_map,p)
            NSE_actors.append(nse_value)
            initial_total_NSE += nse_value
        
        best_total_NSE =  initial_total_NSE
        for modification in self.modifications:
            if modification not in visited:
                utility, updated_NSE, NSE_arr = self.utility_multiple_actors(modification,policies, initial_total_NSE)
                if utility > best_util:
                    best_modification = modification
                    best_util = utility
                    best_total_NSE = updated_NSE
                    NSE_actors = NSE_arr
                
        print("best modification = %s, utility= %s, total NSE = %s"%(best_modification, best_util, sum(NSE_actors)))
        return best_util, best_modification, NSE_actors


    def utility_multiple_actors(self,modification, policies, current_NSE):
        updated_map = self.modifications[modification]
        updated_total_NSE = 0
        NSE_actors = []
        for p in policies:
            nse_value = self.getNSE(updated_map, p)
            updated_total_NSE += nse_value
            NSE_actors.append(nse_value)
        utility = (current_NSE - updated_total_NSE) - self.costs_modification[modification]
        return float(utility), updated_total_NSE, list(NSE_actors)


    def utility(self,modification, policy):
        updated_map = self.modifications[modification]
        updated_NSE = self.getNSE(updated_map, policy)
        utility = (self.initial_NSE - updated_NSE) - self.costs_modification[modification]
        return float(utility), updated_NSE

    def distance(self,mod1, mod2):
        grid1 = readmap(self.modifications[mod1])
        grid2 = readmap(self.modifications[mod2])
        match = 0
        total = 0

        for row in range(len(grid1)):
            x = grid1[row]
            y = grid2[row]
            total += len(x)
            match += np.sum(x == y)

        distance = 1 - (match * 1.0 /total)
        return distance

    def cluster_best_design(self,policy,k=3,actors=1):
        candidate_mod = list(self.modifications.keys())
        visited = []
        for m1 in self.modifications:
            for m2 in self.modifications:
                if m1 != m2:
                    dist = self.distance(m1,m2)
                    if dist <= 0.1:
                        cost1 = self.costs_modification[m1]
                        cost2 = self.costs_modification[m2]
                        if m1 < m2:
                            if m2 in candidate_mod:
                                candidate_mod.remove(m2)
                                visited.append(m2)

                        else:
                            if m1 in candidate_mod:
                                candidate_mod.remove(m1)
                                visited.append(m1)

                        if len(candidate_mod) == k:
                            print("Candidate modifications: ", candidate_mod)
                            if "null" in visited:
                                visited.remove("null")
                            return visited   
        
        return visited
