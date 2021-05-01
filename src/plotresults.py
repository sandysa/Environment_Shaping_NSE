##########################################################################
# Author: Sandhya Saisubramanian
# Description: Generates plots for single actor, single designer setting
##########################################################################
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import seaborn as sns
from collections import OrderedDict

def readDesignResults(filename):
	f = open(filename,"r")
	avg = []
	std= []
	shaping_budget_NSE = []
	shaping_budget_std_NSE = []
	costs = []
	shaping_budget_costs = []
	budget = []
	baseline_NSE = []
	baseline_costs = []
	std_baseline_nse = []

	for line in f:
		if "Budget" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				budget.append(v)
		if "Shaping_Average NSE" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					avg.append(float(v))
		if "Shaping_budget_NSE" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					shaping_budget_NSE.append(float(v))
		if "Shaping_budget_Std_nse" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					shaping_budget_std_NSE.append(float(v))

		if "Shaping_Std_dev_nse" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					std.append(float(v))
		
		if "Baseline_NSE" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					baseline_NSE.append(float(v))

		if "Std_dev_baseline_nse" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					std_baseline_nse.append(float(v))

		if "Baseline_costs" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					baseline_costs.append(float(v))

		if "Shaping_Average cost" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					costs.append(float(v))
		if "Shaping_budget_cost" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					shaping_budget_costs.append(float(v))

	f.close()

	return budget, avg,std, costs, baseline_NSE, std_baseline_nse, baseline_costs, shaping_budget_costs,\
			shaping_budget_NSE,shaping_budget_std_NSE

def readFeedbackResults(filename):
	f = open(filename,"r")
	avg = []
	std= []
	costs = []
	for line in f:
		if "Average" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					if(float(v) > 100):
						avg.append(100)
					else:
						avg.append(float(v))
		if "Std_dev_NSE" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					std.append(float(v))
		if "Actor costs" in line:
			temp = line.strip().replace("[","").replace("]","").split("=")
			val = temp[1].strip().split(",")
			for v in val:
				if v!= "":
					if(float(v) > 100):
						costs.append(100)
					else:
						costs.append(float(v))
	f.close()
	return avg, std, costs


design_filename = sys.argv[1]
feedback_filename = sys.argv[2]
gen_feedback_filename = sys.argv[3]
op_file = sys.argv[4]

budget, avg,std, costs,baseline_NSE,\
 std_baseline_nse, baseline_costs, shaping_budget_costs,\
			shaping_budget_NSE,shaping_budget_std_NSE = readDesignResults(design_filename)

feedback_nse, feedback_std, feedback_costs = readFeedbackResults(feedback_filename)
gen_nse, gen_std, gen_costs = readFeedbackResults(gen_feedback_filename)


# Converting standard deviation to standard error for plots
std = np.array(std)/10
feedback_std = np.array(feedback_std)/10
gen_std = np.array(gen_std)/10
std_baseline_nse = np.array(std_baseline_nse)/10
shaping_budget_std_NSE = np.array(shaping_budget_std_NSE)/10


N = np.arange(len(budget))
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
plt.ylabel("Average NSE penalty",fontsize='14')
plt.xlabel("#Observed actor trajectories",fontsize='14')
ax.set_xticks(N)
ax.set_xticklabels(budget,fontsize='14')
plt.yticks(fontsize='16')
plt.plot(baseline_NSE,linewidth=2, linestyle="-", color="green",marker="p", label="Initial")
plt.plot(feedback_nse,linewidth=2, linestyle="-.", color="red",marker="*", label="Feedback")
plt.plot(gen_nse,linewidth=2, linestyle="-.", color="brown",marker="p", label="Feedback w/ generalization")
plt.plot(avg,linewidth=2, linestyle="--", color="blue",marker="o", label="Shaping")
plt.plot(shaping_budget_NSE,linewidth=2, linestyle="--", color="purple",marker="+", label="Shaping w/ budget")


ax.fill_between(N,np.array(gen_nse)+ np.array(gen_std), np.array(gen_nse)-np.array(gen_std), color='peachpuff')
ax.fill_between(N,np.array(baseline_NSE)+ np.array(std_baseline_nse), np.array(baseline_NSE)-np.array(std_baseline_nse), color='lightgreen')
ax.fill_between(N,np.array(avg)+ np.array(std), np.array(avg)-np.array(std), color='cyan')
ax.fill_between(N,np.array(shaping_budget_NSE)+ np.array(shaping_budget_std_NSE), np.array(shaping_budget_NSE)-np.array(shaping_budget_std_NSE), color='pink')
ax.fill_between(N,np.array(feedback_nse)+ np.array(feedback_std), np.array(feedback_nse)-np.array(feedback_std), color='salmon')
plt.legend(fontsize='14')
box = ax.get_position()
ax.legend(loc ='upper left',ncol=3,handletextpad=0.1,columnspacing=0.8,bbox_to_anchor=(-0.12, box.height+0.5),fancybox=False,fontsize='14')
plt.savefig(op_file+"_trials_NSE.png",bbox_inches='tight')


# Plot actor's expected costs
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
plt.ylabel("Cost",fontsize='14')
plt.xlabel("#Observed actor trajectories",fontsize='14')
ax.set_xticks(N)
ax.set_xticklabels(budget,fontsize='14')
plt.yticks(fontsize='14')
plt.plot(baseline_costs,linewidth=3, linestyle="-", color="green",marker="p", label="Initial")
plt.plot(gen_costs,linewidth=3, linestyle="-.", color="brown",marker="p", label="Feedback w/ generalization")
plt.plot(costs,linewidth=3, linestyle="--", color="blue",marker="o", label="Shaping")
plt.plot(feedback_costs,linewidth=3, linestyle="-.", color="red",marker="*", label="Feedback")

plt.plot(shaping_budget_costs,linewidth=3, linestyle="--", color="purple",marker="+", label="Shaping w/ Budget")

plt.legend(fontsize='14')
box = ax.get_position()
ax.legend(loc ='upper left',ncol=3,handletextpad=0.1,columnspacing=0.8,bbox_to_anchor=(-0.12, box.height+0.5),fancybox=False,fontsize='14')
plt.savefig(op_file+"_trials_primary.png",bbox_inches='tight')
