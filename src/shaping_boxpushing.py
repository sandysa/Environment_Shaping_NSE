########################################################################################
# Author: Sandhya Saisubramanian
# Description: Implements environment shaping and actor-designer coordination
# 			   for boxpushing domain (single actor setting)
########################################################################################
import numpy as np
import sys
import os 
import time

from boxpushing import Boxpushing 
from designer import Designer
from domain_helper import get_modifications

current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_path, '..'))

BOXPUSHING_MAP_PATH = os.path.join(current_file_path, '..', 'maps','boxpushing')

def getE0(domain_name, avoidable):
	if domain_name == "boxpushing":
		if avoidable == True:
			return os.path.join(BOXPUSHING_MAP_PATH,'grid-1.bp')
		else:
			return os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable.bp')

def setup_boxpushing():
	maps = [os.path.join(BOXPUSHING_MAP_PATH,f) for f in os.listdir(BOXPUSHING_MAP_PATH) if os.path.isfile(os.path.join(BOXPUSHING_MAP_PATH,f))]
	if avoidable_NSE == False:
		E0 = os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable.bp')
		
		modifications = {
			'null':  E0,
			'moveVase_bottomleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_vase_bottomleft.bp'),
			'moveVase_topleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_vase_topleft.bp'),
			'moveVase_bottomright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_vase_bottomright.bp'),
			'moveVase_topright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_vase_topright.bp'),
			'remove_rug': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug.bp'),
			'remove_rug_remove_vase': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_remove_vase.bp'),
			'block':  os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_block.bp'),
			'block_rug':  os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_block_rug.bp'),
			'remove_vase': os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_remove_vase.bp'),
			'block_rug_moveVase_bottomleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_block_rug_vase_bottomleft.bp'),
			'block_rug_moveVase_bottomright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_block_rug_vase_bottomright.bp'),
			'block_rug_moveVase_topleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_block_rug_vase_topleft.bp'),
			'block_rug_moveVase_topright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1-unavoidable_block_rug_vase_topright.bp'),
			'remove_rug_moveVase_bottomleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_bottomleft.bp'),
			'remove_rug_moveVase_bottomright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_bottomright.bp'),
			'remove_rug_moveVase_topleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_topleft.bp'),
			'remove_rug_moveVase_topright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_topright.bp'),
			'add_sheet':  os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug.bp'),
			'add_sheet_moveVase_bottomleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_bottomleft.bp'),
			'add_sheet_moveVase_bottomright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_bottomright.bp'),
			'add_sheet_moveVase_topleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_topleft.bp'),
			'add_sheet_moveVase_topright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_topright.bp'),
			'add_sheet_remove_vase': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_remove_vase.bp')									
			}

		cost_modifications = {
			'null': 0,
			'moveVase_bottomleft': 1,
			'moveVase_topleft': 1,
			'moveVase_topright': 1,
			'moveVase_bottomright': 1,
			'remove_rug': 42,  #0.4/unit
			'remove_rug_remove_vase': 44, #0.4/unit+2
			'add_sheet': 21,  #0.2/unit
			'add_sheet_remove_vase': 13,
			'block': 22, 
			'block_rug': 21, #0.2/unit
			'remove_vase': 2,
			'block_rug_moveVase_bottomleft':22, 
			'block_rug_moveVase_bottomright': 22,
			'block_rug_moveVase_topright': 22,
			'block_rug_moveVase_topleft': 22,
			'remove_rug_moveVase_bottomleft': 45,
			'remove_rug_moveVase_bottomright': 45,
			'remove_rug_moveVase_topleft': 45,
			'remove_rug_moveVase_topright': 45,
			'add_sheet_moveVase_bottomleft': 22,
			'add_sheet_moveVase_bottomright': 22,
			'add_sheet_moveVase_topleft': 22,
			'add_sheet_moveVase_topright': 22
			}  

	else:
		E0 = os.path.join(BOXPUSHING_MAP_PATH,'grid-1.bp')
		modifications = {
			'null':  E0,
			'moveVase_bottomleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_vase_bottomleft.bp'),
			'moveVase_topleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_vase_topleft.bp'),
			'moveVase_bottomright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_vase_bottomright.bp'),
			'moveVase_topright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_vase_topright.bp'),
			'remove_rug': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug.bp'),
			'remove_rug_remove_vase': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_remove_vase.bp'),
			'block':  os.path.join(BOXPUSHING_MAP_PATH,'grid-1_block.bp'),
			'block_rug':  os.path.join(BOXPUSHING_MAP_PATH,'grid-1_block_rug.bp'),
			'remove_vase': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_remove_vase.bp'),
			'block_rug_moveVase_bottomleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_block_rug_vase_bottomleft.bp'),
			'block_rug_moveVase_bottomright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_block_rug_vase_bottomright.bp'),
			'block_rug_moveVase_topleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_block_rug_vase_topleft.bp'),
			'block_rug_moveVase_topright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_block_rug_vase_topright.bp'),
			'remove_rug_moveVase_bottomleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_bottomleft.bp'),
			'remove_rug_moveVase_bottomright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_bottomright.bp'),
			'remove_rug_moveVase_topleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_topleft.bp'),
			'remove_rug_moveVase_topright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_topright.bp'),
			'add_sheet':  os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug.bp'),
			'add_sheet_moveVase_bottomleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_bottomleft.bp'),
			'add_sheet_moveVase_bottomright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_bottomright.bp'),
			'add_sheet_moveVase_topleft': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_topleft.bp'),
			'add_sheet_moveVase_topright': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_vase_topright.bp'),
			'add_sheet_remove_vase': os.path.join(BOXPUSHING_MAP_PATH,'grid-1_norug_remove_vase.bp')									
			}

		# Cost of each modification proportional to carpet area in E0
		# Alternatively, this can be defined as per unit cost and 
		# the exact value can be extracted for each configuration.
		cost_modifications = {
			'null': 0,
			'moveVase_bottomleft': 1,
			'moveVase_topleft': 1,
			'moveVase_topright': 1,
			'moveVase_bottomright': 1,
			'remove_rug': 18,  #0.4/unit
			'remove_rug_remove_vase': 20, #0.4/unit+2
			'add_sheet': 9,  #0.2/unit
			'add_sheet_remove_vase': 11,
			'block': 10, 
			'block_rug': 9, #0.2/unit
			'remove_vase': 2,
			'block_rug_moveVase_bottomleft':10, 
			'block_rug_moveVase_bottomright': 10,
			'block_rug_moveVase_topright': 10,
			'block_rug_moveVase_topleft': 10,
			'remove_rug_moveVase_bottomleft': 19,
			'remove_rug_moveVase_bottomright': 19,
			'remove_rug_moveVase_topleft': 19,
			'remove_rug_moveVase_topright': 19,
			'add_sheet_moveVase_bottomleft': 10,
			'add_sheet_moveVase_bottomright': 10,
			'add_sheet_moveVase_topleft': 10,
			'add_sheet_moveVase_topright': 10
			}  
	
	delta_designer  = 0
	delta_actor  = 0

	bp = Boxpushing(E0)
	bp.populateParam()
	states = bp.getStates()
	return maps, modifications, cost_modifications, E0, states, delta_designer, delta_actor


def actor_boxpushing(filename,trajectory_budget=-1):
	bp = Boxpushing(filename)
	if trajectory_budget > -1:
		return bp.solve_simulate(trajectory_budget)
	return bp.solve()

def actor_final_NSE_boxpushing(filename,policy):
	bp = Boxpushing(filename)
	return bp.calculate_final_cost_NSE(policy)


def solve_boxpushing(trajectory_budget_list, cluster=True):
	k = 10000 # NSE penalty when the actor is unable to reach the goal
	
	trajectory_budget_modifications = []
	trajectory_budget_NSE = []
	trajectory_budget_std_NSE = []
	trajectory_budget_costs = []
	trajectory_budget_std_costs = []

	maps, modifications, cost_modifications, E0, states, delta_designer, delta_actor_percentage = setup_boxpushing()
	for trajectory_budget in trajectory_budget_list:
		visited = []
		best_util = 0
		best_modification = 'null'
		num_tested = 0

		policy,visitation_freq, expected_cost = actor_boxpushing(E0,trajectory_budget)
		updated_cost = expected_cost
		designer = Designer(maps, modifications, cost_modifications, E0, delta_designer, domain_name, states)
		designer.populateParameters(policy,visitation_freq)
		current_NSE  = designer.getNSE(E0, policy)
		delta_actor = delta_actor_percentage * expected_cost

		best_config = E0
		best_NSE = current_NSE
		iter = 0
		start = time.time()
		
		# Computes modifications that do not require testing since they are too 
		# similar to other modifications with better utility.
		if(cluster):
			print("Shaping budget = ",shaping_budget)
			visited = designer.cluster_best_design(policy,shaping_budget)

		start1 = time.time()
		while current_NSE > delta_designer:
			if len(visited) == len(modifications):
				break
			iter += 1
			utility, modification, updated_NSE = designer.best_design(visited, policy)
			visited.append(modification)
			num_tested += 1
			if modification == 'null':
				break
			updated_policy, visitation_freq, updated_cost = actor_boxpushing(modifications[modification],trajectory_budget)
			if  updated_cost - expected_cost > delta_actor:
				updated_NSE = k
			else:
				policy = updated_policy
			current_NSE = updated_NSE

			if updated_NSE < best_NSE:
				best_util = utility
				best_modification = modification
				best_config = modifications[modification]
				best_NSE = updated_NSE

		print("***************************************************************************")
		print("Num_tested = %s time_taken(incl. similarity calculation) = %s time taken = %s\n"%(num_tested, time.time()-start, time.time()-start1))
		print("Best modification after shaping = ", best_modification)
		policy,expected_cost = actor_boxpushing(modifications[best_modification])
		avg_cost, std_cost, avg_nse, std_nse = actor_final_NSE_boxpushing(modifications[best_modification],policy)
		trajectory_budget_modifications.append(best_modification) 
		trajectory_budget_NSE.append(avg_nse) 
		trajectory_budget_std_NSE.append(std_nse)
		trajectory_budget_costs.append(expected_cost)
		trajectory_budget_std_costs.append(0)

	return trajectory_budget_modifications, trajectory_budget_NSE, trajectory_budget_std_NSE,\
				trajectory_budget_costs, trajectory_budget_std_costs, delta_designer, delta_actor

def noShaping_boxpushing():
	E0 = getE0("boxpushing",avoidable_NSE)
	policy,expected_cost = actor_boxpushing(E0)
	noshaping_avg_cost, noshaping_std_cost, noshaping_avg_nse, noshaping_std_nse = actor_final_NSE_boxpushing(E0,policy)

	# return noshaping_avg_nse, noshaping_std_nse, noshaping_avg_cost, noshaping_std_cost
	return noshaping_avg_nse, noshaping_std_nse, expected_cost, 0

def main():
	cluster_arr_NSE = []
	cluster_arr_std_nse =[]
	cluster_arr_cost = []
	cluster_arr_std_cost = []
	arr_NSE = []
	arr_std_nse =[]
	arr_cost = []
	arr_std_cost = []

	delta_designer = -1
	delta_actor = -1
	best_modification = []
	trajectory_budget_list = [2,5,10,20,50,100]

	op_file = ""
	if domain_name == "boxpushing":
		op_file = "../results/bp_trials.txt"
		if avoidable_NSE == False:
			op_file = "../results/bp_unavoidable_trials.txt"

		
		cluster_best_modification, cluster_arr_NSE, cluster_arr_std_nse, cluster_arr_cost, cluster_arr_std_cost,\
			  delta_designer, delta_actor = solve_boxpushing(trajectory_budget_list)

		print("Shaping with exhaustive search...\n")
		best_modification, arr_NSE, arr_std_nse, arr_cost, \
		arr_std_cost,delta_designer, delta_actor = solve_boxpushing(trajectory_budget_list,False)

		noshaping_avg_nse,\
			 noshaping_std_nse, noshaping_avg_cost, noshaping_std_cost = noShaping_boxpushing()

		arr_noshaping_NSE = [noshaping_avg_nse for i in trajectory_budget_list]
		arr_noshaping_std_NSE = [noshaping_std_nse for i in trajectory_budget_list]
		arr_noshaping_costs = [noshaping_avg_cost for i in trajectory_budget_list]
		arr_noshaping_std_costs = [noshaping_std_cost for i in trajectory_budget_list]
	
	with open(op_file, 'w+') as f:
		f.write("Budget=%s\n"%trajectory_budget_list)
		f.write("delta_actor=%s\n"%delta_actor)
		f.write("delta_designer=%s\n"%delta_designer)

		f.write("Baseline_NSE=%s\n"%arr_noshaping_NSE)
		f.write("Std_dev_baseline_nse=%s\n"%arr_noshaping_std_NSE)
		f.write("Baseline_costs=%s\n"%arr_noshaping_costs)
		
		f.write("Shaping_budget_NSE=%s\n"%cluster_arr_NSE)
		f.write("Shaping_budget_Std_nse=%s\n"%cluster_arr_std_nse)
		f.write("Shaping_budget_cost=%s\n"%cluster_arr_cost)

		f.write("Shaping_budget_modifications=%s\n"%cluster_best_modification)

		f.write("Shaping_Average NSE=%s\n"%arr_NSE)
		f.write("Shaping_Std_dev_nse=%s\n"%arr_std_nse)
		f.write("Shaping_Average cost=%s\n"%arr_cost)
		f.write("Shaping_Best modifications=%s\n"%best_modification)
	

if __name__ == '__main__':
	avoidable_NSE = True
	shaping_budget = 3
	domain_name = "boxpushing"
	if len(sys.argv) > 1:
		if sys.argv[1] == "unavoidable":
			avoidable_NSE = False
	main()
