########################################################################################
# Author: Sandhya Saisubramanian
# Description: Implements environment shaping and actor-designer coordination
# 			   for driving domain (multiple actor setting)
########################################################################################
import numpy as np
import sys
import os 
import random
import time

from driving import Driving
from designer import Designer
from domain_helper import *
from sklearn.ensemble import RandomForestClassifier

current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_path, '..'))
DRIVING_MAP_PATH = os.path.join(current_file_path, '..', 'maps','driving')

random.seed(100)

def Predict(x_train,y_train,x_test):
	final_model = RandomForestClassifier(n_estimators=100)
	x=np.array(x_train)
	y=np.array(y_train)
	final_model.fit(x_train,y_train)

	test_label = final_model.predict(x_test)

	return test_label

def actor_final_NSE_driving(map, start, goal,policy):
	dr = Driving(map,start,goal)
	return dr.calculate_final_cost_NSE(policy)

def actor_driving(map, start, goal,trajectory_budget=-1):
	dr = Driving(map,start,goal)
	if trajectory_budget > -1:
		return dr.solve_simulate(trajectory_budget)
	return dr.solve()

def solve_disapproved_actions(filename,start,goal,disapproved_actions={}):
	dr = Driving(filename,start,goal)
	return dr.solve_feedback(disapproved_actions)

def getE0():
	E0 = os.path.join(DRIVING_MAP_PATH,'grid-1.dr')
	return E0

def setup_driving():
	start_locations = []
	goal_locations = []
	maps = [os.path.join(DRIVING_MAP_PATH,f) for f in os.listdir(DRIVING_MAP_PATH) if os.path.isfile(os.path.join(DRIVING_MAP_PATH,f))]
	E0 = os.path.join(DRIVING_MAP_PATH,'grid-1.dr')
	modifications = {
			'null':  E0,
			'reduce_speed_all': os.path.join(DRIVING_MAP_PATH,'grid-1_reduced_speed.dr'),
			'fill_potholes': os.path.join(DRIVING_MAP_PATH,'grid-1_fill.dr'),
			'fill_deep_potholes': os.path.join(DRIVING_MAP_PATH,'grid-1_fill_deep.dr'),
			'reduce_speed_fill_deep': os.path.join(DRIVING_MAP_PATH,'grid-1_fill_reduce.dr'), #reduces speed at all shallow potholes and fills deep potholes
			'reduce_speed_zone1': os.path.join(DRIVING_MAP_PATH,'grid-1_reduced_speed1.dr'),
			'reduce_speed_zone2': os.path.join(DRIVING_MAP_PATH,'grid-1_reduced_speed2.dr'),
			'reduce_speed_zone3': os.path.join(DRIVING_MAP_PATH,'grid-1_reduced_speed3.dr'),
			'reduce_speed_zone4': os.path.join(DRIVING_MAP_PATH,'grid-1_reduced_speed4.dr')
			}

	# Cost of each modification proportional to pothole area in E0
	# Alternatively, this can be defined as per unit cost and 
	# the exact value can be extracted for each configuration.
	cost_modifications = {
		'null': 0,
		'reduce_speed_zone1': 10,
		'reduce_speed_zone2': 16,
		'reduce_speed_zone3': 30,
		'reduce_speed_zone4': 28,
		'reduce_speed_all': 84, #cost/unit=2
		'fill_potholes': 168, #cost/unit=4
		'fill_deep_potholes': 76,
		'reduce_speed_fill_deep': 122
		}

	dummy_start = (1,1)
	dummy_goal = (13,22)
	dr = Driving(E0,dummy_start, dummy_goal)
	states = dr.getStates()

	for a in range(number_actors):
		s,g = get_start_goal_driving(states)
		start_locations.append(s)
		goal_locations.append(g)

	return maps, modifications, cost_modifications, E0, states, start_locations,goal_locations

def solve_driving(trajectory_budget_list, start_locations=[], goal_locations=[],cluster=False):
	k = 10000 # NSE penalty when the actor is unable to reach the goal
	trajectory_budget_modifications = []
	trajectory_budget_NSE = []
	trajectory_budget_costs = []

	maps, modifications, cost_modifications, E0, states,start_loc,goal_loc = setup_driving()
	if start_locations == []:
		start_locations = start_loc
		goal_locations = goal_loc

	for trajectory_budget in trajectory_budget_list:
		visited = []
		delta_actor_arr = []
		curr_policies = []
		current_NSE = []

		best_total_nse = 0
		violation = False
		best_modification = 'null'
		expected_costs = []
		nse_val = 0
		num_tested = 0

		designer = Designer(maps, modifications, cost_modifications, E0, delta_designer_percentage, domain_name, states)
		
		start = time.time()
		for actor in range(number_actors):
			policy,visitation_freq, expected_cost = actor_driving(E0, start_locations[actor], goal_locations[actor],trajectory_budget)
			curr_policies.append(policy)
			delta_actor = delta_actor_percentage * expected_cost
			delta_actor_arr.append(delta_actor)
			expected_costs.append(expected_cost)
			
			designer.populateParameters(policy,visitation_freq)
			nse_val += designer.getNSE(E0, policy)
			current_NSE.append(nse_val)
		
		# Computes modifications that do not require testing since they are too 
		# similar to other modifications with better utility.
		if(cluster):
			print("Shaping budget = ",shaping_budget)
			visited = designer.cluster_best_design(curr_policies,shaping_budget,number_actors)

		start1 = time.time()

		delta_designer = delta_designer_percentage * nse_val
		if nse_val > delta_designer:
			violation = True
		best_total_nse = nse_val
		
		while violation:
			violation = False
			current_NSE = []
			utility, modification, updated_NSE_actors = designer.best_design_multiple_actors(visited, curr_policies)
			visited.append(modification)
			num_tested += 1
			if modification == 'null':
				break
			
			
			for actor in range(number_actors):
				updated_policy, visitation_freq, updated_cost = actor_driving(modifications[modification], start_locations[actor],\
																			 goal_locations[actor],trajectory_budget)
				
				if  updated_cost - expected_costs[actor] > delta_actor_arr[actor]:
					updated_NSE_actors[actor] = k	
				else:
					policy = updated_policy

			if sum(updated_NSE_actors) > delta_designer:
				violation = True

			current_NSE = updated_NSE_actors

			if sum(current_NSE) < best_total_nse:
				best_modification = modification
				best_config = modifications[modification]
				best_NSE = current_NSE

		print("***************************************************************************")
		print("Time taken (s) =%s time taken (wo similarity calculation) = %s num_tested=%s\n"%((time.time() -start),(time.time()-start1),num_tested))
		print("Best modification = %s\n"%best_modification)
		expected_costs = []
		final_nse = []
		trajectory_budget_modifications.append(best_modification)
		for actor in range(number_actors):
			policy, expected_cost = actor_driving(modifications[best_modification], start_locations[actor], goal_locations[actor])
			expected_costs.append(expected_cost)
			avg_cost, std_cost, avg_nse, std_nse = actor_final_NSE_driving(modifications[best_modification],\
														start_locations[actor], goal_locations[actor],policy)
			final_nse.append(avg_nse)
		
		trajectory_budget_NSE.append(final_nse)
		trajectory_budget_costs.append(expected_costs)

	return trajectory_budget_NSE, trajectory_budget_costs, trajectory_budget_modifications, start_locations, goal_locations


def get_start_goal_driving(states):
	E0 = os.path.join(DRIVING_MAP_PATH,'grid-1.dr')
	grid = readmap(E0)
	walls = WallLoc_driving(grid)
	valid_state = False
	while(valid_state == False):
		start_state = random.choice(states)
		goal_state = random.choice(states)
		if start_state and goal_state not in walls:
			valid_state = True
	return start_state, goal_state

def noShaping_driving(start_locations, goal_locations):
	baseline_nse = []
	baseline_costs = []

	E0 = os.path.join(DRIVING_MAP_PATH,'grid-1.dr')
	for actor in range(number_actors):
		policy,expected_cost = actor_driving(E0, start_locations[actor], goal_locations[actor])
		avg_cost, std_cost, avg_nse, std_nse = actor_final_NSE_driving(E0, start_locations[actor], \
																		goal_locations[actor],policy)
		baseline_nse.append(avg_nse)
		baseline_costs.append(expected_cost)

	baseline_avg_nse = sum(baseline_nse)/number_actors
	baseline_std_nse = np.std(np.array(baseline_nse))
	baseline_avg_cost = sum(baseline_costs)/number_actors

	return baseline_avg_nse, baseline_std_nse, baseline_avg_cost

def DisapprovedActions(approval,all_states,sa_list,all_actions,generalize_feedback=False):
	x_train = []
	y_train = []
	x_test = []
	testing_sa = []
	disapproved_actions = {s:[] for s,state in enumerate(all_states)}
	# Based on gathered feedback
	for info in approval:
		temp=[]
		s = info[0]
		temp.append(s[0])
		temp.append(s[1])
		temp.append(all_actions.index(info[1]))
		x_train.append(temp)
		y_train.append(int(info[2]))
		
		state_index = all_states.index(info[0])
		if int(info[2]) == 0:
			disapproved_actions[state_index].append(info[1])

	# Generalize the gathered data to unseen states:	
	if generalize_feedback == True:
		for sa in sa_list:
			sa_val = sa[1:]
			temp=[]
			if sa_val not in x_train:
				s = info[0]
				temp.append(s[0])
				temp.append(s[1])
				temp.append(all_actions.index(sa_val[1]))
				x_test.append(temp)	
				testing_sa.append(sa_val)

		y_label = Predict(x_train, y_train, x_test)

		for i in range(len(y_label)):
			if y_label[i] == 0:
				sa_val = testing_sa[i]
				state_index = all_states.index(sa_val[0])
				disapproved_actions[state_index].append(sa_val[1])

	return disapproved_actions

def feedback(trajectory_budget_list,start_locations, goal_locations,generalize_feedback=False):
	feedback_budget = 500
	E0 = getE0()

	grid = readmap(E0)
	dr = Driving(E0,start_locations[0], goal_locations[0])
	all_states = dr.getStates()
	all_actions = dr.getActions()
	sa_list = dr.generate_state_actions()
	NSE_locations = PotholeLoc_driving(grid)
	NSE_values = []
	std_NSE = 0

	for trajectory_budget in trajectory_budget_list:
		total_nse = []
		for actor in range(number_actors):
			policy,visitation_freq, expected_cost = actor_driving(E0, start_locations[actor], goal_locations[actor],trajectory_budget)
			nse_penalty = NSE_penalty_driving(all_states, policy,NSE_locations,visitation_freq)
			delta_designer = delta_designer_percentage * nse_penalty
			feedback_count = 0
		
			approval = []
			if nse_penalty > delta_designer:
				for s,state in enumerate(all_states):
					if feedback_count >= feedback_budget:
						break
					if s in policy:
						temp = []
						action = policy[s]
						if mild_NSE_driving(state,action, NSE_locations) or severe_NSE_driving(state,action, NSE_locations):			
							temp.append(state)
							temp.append(action)
							temp.append(0)
						else:
							temp.append(state)
							temp.append(action)
							temp.append(1)
						approval.append(temp)
						feedback_count += 1
				disapproved_actions = DisapprovedActions(approval,all_states,sa_list,all_actions,generalize_feedback)
				updated_policy,updated_expected_cost = solve_disapproved_actions(E0,start_locations[actor], \
																			goal_locations[actor],disapproved_actions)
				if updated_policy != None and updated_expected_cost - expected_cost <= delta_actor_percentage * expected_cost:
					avg_cost, std_cost, avg_nse, std_nse = actor_final_NSE_driving(E0, start_locations[actor], \
																			goal_locations[actor],updated_policy)
			
					total_nse.append(avg_nse)
				else:
					policy,expected_cost = actor_driving(E0, start_locations[actor], goal_locations[actor])
					avg_cost, std_cost, avg_nse, std_nse = actor_final_NSE_driving(E0, start_locations[actor], \
																			goal_locations[actor],policy)
					total_nse.append(avg_nse)
			else:
				policy,expected_cost = actor_driving(E0, start_locations[actor], goal_locations[actor])
				avg_cost, std_cost, avg_nse, std_nse = actor_final_NSE_driving(E0, start_locations[actor], \
																			goal_locations[actor],policy)
			
				total_nse.append(avg_nse)
			std_NSE = np.std(np.array(total_nse))
		NSE_values.append(sum(total_nse)/number_actors)

	return NSE_values, std_NSE



def main():
	avg_nse = []
	std_nse = []
	avg_cost = []

	shaping_budget_avg_nse = []
	shaping_budget_std_nse = []
	shaping_budget_avg_cost = []

	trajectory_budget_list = [100]

	op_file = "../results/multiple_actors_driving_trials.txt"
	print("Shaping with exhaustive search..\n")
	arr_NSE, arr_cost,best_modification, start_locations, goal_locations = solve_driving(trajectory_budget_list)


	print("Shaping with budget..\n")
	cluster_arr_NSE, cluster_arr_cost,cluster_best_modification, \
	start_locations, goal_locations = solve_driving(trajectory_budget_list,start_locations, goal_locations,True)

	
	baseline_avg_nse, baseline_std_nse, baseline_avg_cost = noShaping_driving(start_locations, goal_locations)
	
	feedback_nse, feedback_std_nse = feedback(trajectory_budget_list,start_locations, goal_locations)

	feedback_gen_nse, feedback_gen_std_nse = feedback(trajectory_budget_list,start_locations, goal_locations,True)
			
	for t in range(len(trajectory_budget_list)):
		avg_nse.append(sum(arr_NSE[t])/number_actors)
		std_nse.append(np.std(np.array(arr_NSE[t])))
		avg_cost.append(sum(arr_cost[t])/number_actors)
		
		shaping_budget_avg_nse.append(sum(cluster_arr_NSE[t])/number_actors)
		shaping_budget_std_nse.append(np.std(np.array(cluster_arr_NSE[t])))
		shaping_budget_avg_cost.append(sum(cluster_arr_cost[t])/number_actors)


	file = open(op_file,"a+")
	file.write("#Actors=%s\n"%number_actors)
	file.write("Designer slack percentage=%s\n"%delta_designer_percentage)
	file.write("Actor slack percentage=25\n")
	file.write("Baseline NSE =%s\n"%baseline_avg_nse)
	file.write("Baseline_std_NSE =%s\n"%baseline_std_nse)
	file.write("Baseline costs =%s\n"%baseline_avg_cost)

	file.write("Shaping_NSE=%s\n"%avg_nse)
	file.write("Shaping_std_NSE=%s\n"%std_nse)
	file.write("Shaping_actor_costs=%s\n"%avg_cost)

	file.write("Shaping_budget_NSE=%s\n"%shaping_budget_avg_nse)
	file.write("Shaping_budget_std_NSE=%s\n"%shaping_budget_std_nse)
	file.write("Shaping_budget_actor_costs=%s\n"%shaping_budget_avg_cost)

	file.write("Feedback_NSE=%s\n"%feedback_nse)
	file.write("Feedback_std_NSE=%s\n"%feedback_std_nse)
	file.write("Feedback_gen_NSE=%s\n"%feedback_gen_nse)
	file.write("Feedback_gen_std_NSE=%s\n"%feedback_gen_std_nse)
	file.write("***********************************\n")
	file.close()


if __name__ == '__main__':
	domain_name = "driving"
	shaping_budget = 4
	delta_designer_percentage = 0
	delta_actor_percentage = 0.25
	number_actors = 10
	if len(sys.argv) > 1: 
		number_actors = int(sys.argv[1])
	main()
	
