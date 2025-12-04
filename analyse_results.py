from tqdm import tqdm
import os
import torch
import pickle

import plotly.graph_objects as go
import plotly.colors as pcolors
import numpy as np

from collections import namedtuple

from datasets import load_dataset

from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from nnsight import LanguageModel




#region Mode select

trn_stage = "pretrain"
# trn_stage = "finetune"
run_sel = 1
SAE_type = "BTK"
# SAE_type = "Standard"

#endregion



#region Predefs


if(SAE_type == "BTK"):
	StampedLog = namedtuple("StampedLog",["step","l0","l2","fve","auxk_loss","loss"])
	fields = [
				"step",
				"l0",
				"l2",
				"fve",
				"auxk_loss",
				"loss",]

elif (SAE_type == "Standard"):
	StampedLog = namedtuple("StampedLog",["step","l0","l2","fve","mse","sparsity_loss","loss"])
	fields = [
				'step',
				'l0', 
				'l2', 
				'fve', 
				'mse', 
				'sparsity_loss', 
				'loss',]



if (trn_stage=="pretrain"):
	
	if(SAE_type == "Standard"):
		d_p = f"/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/store/pred_SAEs/pre{run_sel}_SAE_train_9-6-25/"
	elif(SAE_type == "BTK"):
		d_p = f"/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/store/BTK_pred_SAEs/BTK_pre{run_sel}_SAE_train_5-8-25/"
	
	namer = "log_9999.pkl" #Contains all the other ones too
	keys = [200*i for i in range(1,51)]
	keys[-1] = keys[-1] - 1

elif (trn_stage == "finetune"):

	if(SAE_type == "Standard"):
		d_p = f"/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/store/fined_SAEs/fine{run_sel}_SAE_train_16-6-25/"
	elif(SAE_type == "BTK"):
		d_p = f"/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/store/fined_SAEs/BTK_fine{run_sel}_SAE_train_5-8-25/"

	namer = "ft_log_179.pkl"
	keys = [45*i for i in range(1,5)]
	keys[-1] = keys[-1] - 1



steps = keys
ae3_p = d_p + "SAE3_logs/" + namer
ae6_p = d_p + "SAE6_logs/" + namer
ae9_p = d_p + "SAE9_logs/" + namer

#endregion



#region Funcs

def print_loss (ae, field, keys):
	for k in keys:
		print(f"At step {k}, the field {field} is " + str(eval(f"ae[{k}].{field}")))
		print()

#Visualizes a training feature against training steps using Plotly.
def plot_single_feat(ae, field, x_data=steps):
	
	
	ind = {fields[i]:i for i in range(len(fields))} #? Generalised form removes need for if/elif
	y_name = field
	y_data = [ae[k][ind[field]] for k in steps]
	

	# Create a new figure
	fig = go.Figure()

	# Add a scatter trace for the line plot
	fig.add_trace(go.Scatter(
		x=x_data,
		y=y_data,
		mode='lines+markers',
		name=y_name
	))

	# Update the layout for a clean and informative look
	fig.update_layout(
		title=f'{y_name} vs. Training Steps',
		xaxis_title='Training Steps',
		yaxis_title=y_name,
		template='plotly_white'	# Use a clean white template
	)

	# Show the plot
	fig.show()

#Plot all features for a particular ae across all steps
def plot_all_feats (ae, title ,x_data=steps):

	
	#region* Preliminaries
	

	fields = list(ae[x_data[0]]._fields)

	#TODO Fix taking in tensor inputs, this is a hack for now
	# fields.remove("fve")

	if "step" in fields:
		fields.remove("step")

	
	#endregion

	ind = {fields[i]:i+1 for i in range(len(fields))} #? i+1, since we don't want steps

	data = [[ae[k][ind[field]] for k in steps ] for field in fields]

	metrics_dict = dict(zip(fields,data))


	# Create a new figure
	fig = go.Figure()

	# Get a list of high-contrast colors. Plotly's built-in qualitative color scales are great for this.
	colors = pcolors.qualitative.Plotly

	# Loop through the metrics dictionary and add a trace for each one
	for i, (name, data) in enumerate(metrics_dict.items()):
		fig.add_trace(go.Scatter(
			x=steps,
			y=data,
			mode='lines+markers',
			name=name,
			line=dict(color=colors[i % len(colors)]) # Cycle through the color list
		))

	# Update the layout for a clean and informative look
	fig.update_layout(
		title=title,
		xaxis_title='Training Steps',
		yaxis_title='Value',
		template='plotly_white',
		legend_title_text='Metrics'
	)

	# Show the plot
	fig.show()


#endregion



#region Load SAEs

with open(ae3_p,"rb") as f:
	ae3 = pickle.load(f)

with open(ae6_p,"rb") as f:
	ae6 = pickle.load(f)

with open(ae9_p,"rb") as f:
	ae9 = pickle.load(f)

#endregion



#region Main Code

# plot_all_feats(ae3 , "AE3 plot")
plot_all_feats(ae6 , "AE6 plot")
# plot_all_feats(ae9 , "AE9 plot")

print("STOP")
print("STOP")

#endregion





