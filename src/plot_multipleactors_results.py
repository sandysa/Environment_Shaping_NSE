##########################################################################
# Author: Sandhya Saisubramanian
# Description: Generates plots for multiple actors, single designer setting
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict


def plotNSE_actors():
	file = open("../results/multiple_actors_driving_trials.txt","r")
	num_actors = []
	baseline_nse = []
	baseline_std_nse = []
	shaping_nse = []
	shaping_std_nse = []
	shaping_budget_nse = []
	shaping_budget_std_nse = []

	feedback_nse = []
	feedback_std_nse = []
	gen_feedback_nse = []
	gen_feedback_std_nse = []

	for line in file:
		temp = line.strip().split("=")
		if "#Actors" in temp[0]:
			num_actors.append(temp[1].strip())
		if "Baseline NSE" in temp[0]:
			baseline_nse.append(float(temp[1].strip()))
		if "Baseline_std_NSE" in temp[0]:
			baseline_std_nse.append(float(temp[1].strip()))
		if "Shaping_NSE" in temp[0]:
			val = temp[1].strip().replace("[","").replace("]","").split(",")
			shaping_nse.append(float(val[-1]))
		if "Shaping_std_NSE" in temp[0]:
			val = temp[1].strip().replace("[","").replace("]","").split(",")
			shaping_std_nse.append(float(val[-1]))
		if "Feedback_NSE" in temp[0]:
			val = temp[1].strip().replace("[","").replace("]","").split(",")
			feedback_nse.append(float(val[-1]))
		if "Feedback_std_NSE" in temp[0]:
			feedback_std_nse.append(float(temp[1].strip()))
		if "Feedback_gen_NSE" in temp[0]:
			val = temp[1].strip().replace("[","").replace("]","").split(",")
			gen_feedback_nse.append(float(val[-1]))
		if "Feedback_gen_std_NSE" in temp[0]:
			gen_feedback_std_nse.append(float(temp[1].strip()))
		if "Shaping_budget_NSE" in temp[0]:
			val = temp[1].strip().replace("[","").replace("]","").split(",")
			shaping_budget_nse.append(float(val[-1]))
		if "Shaping_budget_std_NSE" in temp[0]:
			val = temp[1].strip().replace("[","").replace("]","").split(",")
			shaping_budget_std_nse.append(float(val[-1]))



	baseline_std_nse = np.array(baseline_std_nse)/10
	shaping_std_nse = np.array(shaping_std_nse)/10
	feedback_std_nse = np.array(feedback_std_nse)/10
	gen_feedback_std_nse = np.array(gen_feedback_std_nse)/10
	shaping_budget_std_nse = np.array(shaping_budget_std_nse)/10

	bar_width = 0.15
	file.close()
	# Get a color map
	my_cmap = plt.get_cmap('tab20c')

	N = np.arange(len(num_actors))
	fig = plt.figure(figsize=(6,4))
	ax = fig.add_subplot(111)
	plt.ylabel("Average NSE penalty",fontsize='14')
	plt.xlabel("#Actors",fontsize='14')
	ax.set_xticks(N+0.3)
	ax.set_xticklabels(num_actors,fontsize='12')
	

	plt.bar(N,baseline_nse,width=bar_width, color=my_cmap(0), label="Initial",yerr=baseline_std_nse,ecolor='black',capsize=3)
	plt.bar(N+bar_width,feedback_nse,width=bar_width,color=my_cmap(0.2),label="Feedback",yerr=feedback_std_nse,ecolor='black',capsize=3,hatch='x')
	plt.bar(N+2*bar_width,gen_feedback_nse,width=bar_width,color=my_cmap(2/5),label="Feedback w/ generalization",yerr=gen_feedback_std_nse,ecolor='black',capsize=3,hatch='-')
	plt.bar(N+3*bar_width,shaping_nse,width=bar_width,color=my_cmap(3/5),label="Shaping",yerr=shaping_std_nse,ecolor='black',capsize=3,hatch='//')
	plt.bar(N+4*bar_width,shaping_budget_nse,width=bar_width,color=my_cmap(4/5),label="Shaping w/ Budget",yerr=shaping_budget_std_nse,ecolor='black',capsize=3,hatch='\\')

	box = ax.get_position()
	ax.legend(loc ='upper left',ncol=2,handletextpad=0.1,columnspacing=0.8,bbox_to_anchor=(-0.12, box.height+0.53),fancybox=False,fontsize='14')
	plt.savefig("../results/"+op_file+"_multiple_actor_shaping_budget.png",bbox_inches='tight')

	# # Plot legend separately
	# figLegend = plt.figure(figsize=(5,3))
	# axi = figLegend.add_subplot(111)
	# handles, labels = ax.get_legend_handles_labels()
	# by_label = OrderedDict(zip(labels, handles))
	# figLegend.legend(handles, labels, loc = 'center',frameon=False,fontsize='15')
	# axi.xaxis.set_visible(False)
	# axi.yaxis.set_visible(False)
	# figLegend.canvas.draw()
	# figLegend.savefig("legend.png")

op_file = sys.argv[1]
plotNSE_actors()