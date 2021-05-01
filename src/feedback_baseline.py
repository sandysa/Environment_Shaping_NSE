#########################################################################
# Author: Sandhya Saisubramanian
# Description: Learning from feedback to update agent policy.
#########################################################################
import numpy as np
import sys
import os 

from boxpushing import Boxpushing 
from domain_helper import *
from sklearn.ensemble import RandomForestClassifier

current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_path, '..'))

BOXPUSHING_MAP_PATH = os.path.join(current_file_path, '..', 'maps','boxpushing')


def Predict(x_train,y_train,x_test):
	final_model = RandomForestClassifier(n_estimators=100)
	x=np.array(x_train)
	y=np.array(y_train)
	final_model.fit(x_train,y_train)

	test_label = final_model.predict(x_test)

	return test_label


def bp_trajectories(filename,trajectory_budget=-1):
	bp = Boxpushing(filename)
	bp.populateParam()
	if trajectory_budget > -1:
		return bp.solve_simulate(trajectory_budget)
	return bp.solve()


def bp_solve_disapproved_actions(filename,disapproved_actions={}):
	bp = Boxpushing(filename)
	bp.populateParam()
	return bp.solve_feedback(disapproved_actions)

def actor_final_NSE_boxpushing(filename,policy):
	bp = Boxpushing(filename)
	bp.populateParam()
	return bp.calculate_final_cost_NSE(policy)


def DisapprovedActions(approval,all_states,sa_list,all_actions):
	x_train = []
	y_train = []
	x_test = []
	training_sa = []
	testing_sa = []
	disapproved_actions = {s:[] for s,state in enumerate(all_states)}
	# Based on gathered feedback
	for info in approval:
		temp = bp_state_to_feature(info[0])
		temp.append(all_actions.index(info[1]))
		x_train.append(temp)
		y_train.append(int(info[2]))
		
		training_sa.append(info[:-1])
		state_index = all_states.index(info[0])
		if int(info[2]) == 0:
			disapproved_actions[state_index].append(info[1])


	# Generalize the gathered data to unseen states:	
	if generalize_feedback == True:
		for sa in sa_list:
			sa_val = sa[1:]
			if sa_val not in x_train:
				temp = bp_state_to_feature(sa_val[0])
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


def solve_boxpushing(feedback_budget,trajectory_budget_list,delta_designer):
	NSE_values = []
	std_NSE_values = []
	actor_costs = []
	std_actor_costs = []

	k = 100
	delta_actor = 0
	E0 = os.path.join(BOXPUSHING_MAP_PATH,'grid-1.bp')
	if avoidable_NSE == False:
		E0 = os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable.bp')

	grid = readmap(E0)
	NSE_locations = CarpetLoc_boxpushing(grid)
	bp = Boxpushing(E0)
	bp.populateParam()
	all_states = bp.getStates()
	all_actions = bp.getActions()
	sa_list = bp.generate_state_actions()

	init_policy, actor_init_cost = bp_trajectories(E0)

	for trajectory_budget in trajectory_budget_list:
		policy,visitation_freq,expected_cost = bp_trajectories(E0,trajectory_budget)
		nse_penalty = NSE_penalty_boxpushing(all_states, policy,NSE_locations,visitation_freq)
		feedback_count = 0
		
		approval = []
		if nse_penalty > delta_designer:
			for s,state in enumerate(all_states):
				if feedback_count >= feedback_budget:
					break
				if s in policy:
					temp = []
					action = policy[s]
					if is_NSE_boxpushing(state,action, NSE_locations):			
						temp.append(state)
						temp.append(action)
						temp.append(0)
					else:
						temp.append(state)
						temp.append(action)
						temp.append(1)
					approval.append(temp)
					feedback_count += 1

			disapproved_actions = DisapprovedActions(approval,all_states,sa_list,all_actions)
			updated_policy,updated_expected_cost = bp_solve_disapproved_actions(E0,disapproved_actions)

			if updated_policy != None and updated_expected_cost - expected_cost <= delta_actor * expected_cost:
				avg_cost, std_cost, avg_nse, std_nse = actor_final_NSE_boxpushing(E0,updated_policy) 
				NSE_values.append(avg_nse)
				std_NSE_values.append(std_nse)
				actor_costs.append(updated_expected_cost)
				std_actor_costs.append(0)

			else:
				policy,expected_cost = bp_trajectories(E0)
				avg_cost, std_cost, avg_nse, std_nse = actor_final_NSE_boxpushing(E0,policy) 
				NSE_values.append(float(nse_penalty))
				std_NSE_values.append(std_nse)
				actor_costs.append(expected_cost)
				std_actor_costs.append(0)
			
		else:
			policy,expected_cost = bp_trajectories(E0)
			avg_cost, std_cost, avg_nse, std_nse = actor_final_NSE_boxpushing(E0,policy) 
			NSE_values.append(avg_nse)
			std_NSE_values.append(std_nse)
			actor_costs.append(expected_cost)
			std_actor_costs.append(std_cost)

		
	return NSE_values, std_NSE_values, actor_costs, std_actor_costs

def main():
	feedback_budget = 500
	delta_designer = 0
	op_file = ""
	trajectory_budget_list = [2,5,10,20,50,100]
	if domain_name == "boxpushing":
		op_file = "../results/bp_feedback_baseline_avoidable.txt"
		if generalize_feedback == True:
			op_file = "../results/bp_feedback_baseline_avoidable_generalize.txt"
		if avoidable_NSE == False:
			op_file = "../results/bp_feedback_baseline_unavoidable.txt"
			if generalize_feedback == True:
				op_file = "../results/bp_feedback_baseline_unavoidable_generalize.txt"

		NSE_values, std_NSE_values, actor_costs, std_actor_costs = solve_boxpushing(feedback_budget,trajectory_budget_list,delta_designer)
		with open(op_file, 'w+') as f:
			f.write("Budget=%s\n"%trajectory_budget_list)
			f.write("Average NSE=%s\n"%NSE_values)
			f.write("Std_dev_NSE=%s\n"%std_NSE_values)
			f.write("Actor costs=%s\n"%actor_costs)
		f.close()
			
		
if __name__ == '__main__':
	avoidable_NSE = True
	generalize_feedback = False
	domain_name = sys.argv[1]
	if len(sys.argv) > 2:
		if sys.argv[2] == "generalize":
			generalize_feedback = True
		if sys.argv[2] == "unavoidable":
			avoidable_NSE = False
	if len(sys.argv) > 3:
		if sys.argv[3] == "unavoidable":
			avoidable_NSE = False

	main()


