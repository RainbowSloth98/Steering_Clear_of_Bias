from tqdm import tqdm
import os
import copy
import torch
import numpy as np
import pickle
from tabulate import tabulate
from collections import defaultdict, Counter
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

from datasets import load_dataset

from dictionary_learning import AutoEncoder
from nnsight import LanguageModel



#region? Info
#? Max_acts store 
#? The keys of the main dict are: 'top_k_values', 'sparse_features', 'metadata'
	#? 1. Top_k_values is a tensor storing the max_activations for all features, shape [feats,topk]/[6144,20]
	#? 2. sparse_features stores the feature distribution on which the specific max_act occurred
	#? 3. metadata stores: 3.1 'value', 3.2'token_id', 3.3'step', 3.4'context'
		#? 3.1 Value represents the value of the feature at that entry
		#? 3.2 token_id is the token that caused that activation to happenb
		#? 3.3 At which step this was taken - mostly irrelevant
		#? 3.4 A radius of 15 words around the token of highest activation, gives context.

#? 2./3. are subdictionaries which store only the non-zero activations. Thier keys are of shape:
# *[sae_idx,feat_idx,k_idx]

#? TODOs:
#TODO Create a way to visualise feature intensity, to see the change as the feat gets stronger.
	#?Do we even have a way of doing that?


#endregion



#region Tune params

#? only pretrained, or the finetuned varient
stage = "pre"
# stage = "fine"

#? Which layers to load, specific up to all three
# to_load = [3,6,9]
to_load = [6]

#pivot
if (stage == "pre"):
	cp = 7000
elif(stage == "fine"):
	cp = 179


#? Distribution visualisation

disvis_thresh = 0.2
disvis_top_n = 50
# fig_type = "heatmap"
fig_type = "bar"



#endregion  



#region Funcs

#Get back N highest features, for K many samples from highest to lowest
def get_n_highest_feats(mact,sae_idx,n,k=5):
	
	
	allfeat_subks = [[(sae_idx,feat_idx,i) for i in range(k)] for feat_idx in range(6144)]

	vals = []

	#Create a list of values associated with the 
	for feat in range(6144):
		vals.append(tuple(mact["metadata"][ent]["value"] for ent in allfeat_subks[feat]))


	#This sorting operation sorts tuples from first to last entry, from highest to lowest.
	s_vals = sorted(vals,reverse=True)

	indicies = []

	for i in range(n):
		indicies.append(vals.index(s_vals[i]))
	
	return (indicies,[vals[i] for i in indicies])


#Get the metadata/sparse_features subkeys - mostly not needed but still used in other funcs
def get_subkeys(max_act):

	if ("metadata" in max_act.keys()):
		return list(max_act["metadata"].keys())
	
	elif ("sparse_features" in max_act.keys()):
		return list(max_act[0]["sparse_features"].keys())

	else:
		raise Exception("No sparse_features or metadata fields.")


#Filter keys by feature
def key_by_feat(key_list, feat_idx):
	return [entry for entry in key_list if entry[1] == feat_idx]


#The main function used to show full individual entries in the SAEs
def show_meta(mact, subdict_key, tokenizer,print=True,): #Input should be a key as used by the dictionaries
	
	dat = copy.deepcopy(mact["metadata"][subdict_key])

	#Decode the token for easy visualisation
	dat["token_id"] = (dat["token_id"],tokenizer.decode(dat["token_id"]))
	


	#region* Highlighting max act token in context

	enc = tokenizer.encode(dat["context"])
	ins = enc.index(dat["token_id"][0]) # Find the index of the token we wish to select

	enc.insert(ins,197) # 197 = |, adding this before the token
	enc.insert(ins,197) # Adding another one
	enc.insert(ins+3,197) #This moves the token 2 ahead, and we want to place in front
	enc.insert(ins+3,197) #So we do 2 additions at +3.

	dat["context"] = tokenizer.decode(enc, skip_special_tokens=True)

	#endregion

	#Create a list of lists to output as.
	table = [[str(k)]+ [dat[k]] for k in list(dat.keys())]

	ret =tabulate(table,tablefmt="github")
	
	if(print):
		print(ret)
		return None

	return ret

#Simple print function
def print_topK(tk):
	for i in tk:
		print(i)



def feat_show_topK (mact, sae_idx ,feat_idx, tokenizer,k=20,print=True):

	ret = []
	subks = [(sae_idx,feat_idx,i) for i in range(0,k)]
	deli = "#"*20

	ret.append(deli)

	for k in subks:
		ret.append(show_meta(mact,k,tokenizer,print=False))

	if print:
		print_topK(ret)
		return None
	
	return ret



#Show the distribution of only the activating features.
def show_sparse_feat_distribution(values, indices, fig_type=fig_type, threshold=disvis_thresh, top_n=disvis_top_n, focus_index=None):

	

	if not isinstance(values, torch.Tensor) or not isinstance(indices, torch.Tensor):
		raise TypeError("Input 'values' and 'indices' must be torch.Tensors.")


	if values.dim() != 1 or indices.dim() != 1 or values.shape != indices.shape:
		raise ValueError("Input tensors must be 1D and of the same shape.")


	# Filter for high activations
	high_mask = values > threshold
	high_values = values[high_mask]
	high_indices = indices[high_mask]

	if high_indices.numel() < 2:
		print("Warning: Less than two values above the threshold. No co-activation to plot.")
		return go.Figure()


	if focus_index is not None:
		# Check if the focus index is active above the threshold
		focus_active_mask = high_indices == focus_index
		if not torch.any(focus_active_mask):
			print(f"Warning: Focus index {focus_index} is not active above the threshold of {threshold}.")
			return go.Figure()

		focus_value = high_values[focus_active_mask].max()

		# Calculate co-activation with the focus index
		coactivations = []
		for i in range(high_indices.numel()):
			partner_idx = high_indices[i]
			if partner_idx != focus_index:
				partner_value = high_values[i]
				strength = focus_value * partner_value
				coactivations.append((partner_idx.item(), strength.item()))
		
		# Remove duplicates, keeping the one with max strength (if any)
		coactivations_dict = {}
		for idx, strength in coactivations:
			if idx not in coactivations_dict or strength > coactivations_dict[idx]:
				coactivations_dict[idx] = strength
		
		sorted_coactivations = sorted(coactivations_dict.items(), key=lambda item: item[1], reverse=True)

		if not sorted_coactivations:
			print(f"Warning: No other features co-activating with index {focus_index} above the threshold.")
			return go.Figure()

		partner_labels = [str(item[0]) for item in sorted_coactivations]
		strengths = [item[1] for item in sorted_coactivations]

		fig = go.Figure([go.Bar(x=partner_labels, y=strengths)])
		fig.update_layout(title_text=f'Co-activations with Feature {focus_index} (Threshold > {threshold})',
						xaxis_title=f"Features Co-activating with Index {focus_index}",
						yaxis_title="Co-activation Strength",
						xaxis={'categoryorder':'total descending'})
		return fig


	if fig_type == 'heatmap':
		# Create a co-activation matrix
		unique_indices = torch.unique(high_indices)
		coactivation_matrix = torch.zeros((unique_indices.size(0), unique_indices.size(0)))

		for i, idx1 in enumerate(unique_indices):
			for j, idx2 in enumerate(unique_indices):
				if i <= j:
					val1 = high_values[high_indices == idx1].max()
					val2 = high_values[high_indices == idx2].max()
					coactivation_matrix[i, j] = val1 * val2
					coactivation_matrix[j, i] = val1 * val2 # Symmetric matrix

		fig = px.imshow(coactivation_matrix,
						x=[str(i.item()) for i in unique_indices],
						y=[str(i.item()) for i in unique_indices],
						labels=dict(x="Index", y="Index", color="Co-activation Strength"),
						title=f"Co-activation Heatmap (Threshold > {threshold})")
		fig.update_xaxes(side="top")

	elif fig_type == 'bar':
		# Calculate pairwise co-activation and get the top N
		coactivations = []
		for i in range(high_indices.numel()):
			for j in range(i + 1, high_indices.numel()):
				idx1 = high_indices[i]
				idx2 = high_indices[j]
				val1 = high_values[i]
				val2 = high_values[j]
				coactivation_strength = val1 * val2
				coactivations.append(((idx1.item(), idx2.item()), coactivation_strength.item()))

		# Sort by co-activation strength
		coactivations.sort(key=lambda x: x[1], reverse=True)
		top_coactivations = coactivations[:top_n]

		if not top_coactivations:
			print("Warning: No co-activations found above the threshold.")
			return go.Figure()

		pair_labels = [f"({pair[0]}, {pair[1]})" for pair, strength in top_coactivations]
		strengths = [strength for pair, strength in top_coactivations]

		fig = go.Figure([go.Bar(x=pair_labels, y=strengths)])
		fig.update_layout(
					title_text=f'Top {top_n} Co-activating Pairs (Threshold > {threshold})',
					xaxis_title="Index Pairs",
					yaxis_title="Co-activation Strength")

	else:
		raise ValueError("Invalid 'fig_type'. Choose either 'heatmap' or 'bar'.")

	return fig




#Similar and much overlap with show_sparse_feat_distribution
def show_act(tens,namer):
	# Create the figure
	fig = go.Figure(data=go.Scatter(y=tens, mode='lines', name='Tensor Values'))

	fig.update_layout(
		title=namer,
		xaxis_title='Index',
		yaxis_title='Value'
	)

	return fig



#endregion



#region Paths and related defs

store_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/store/"

max_act_p = store_p + f"{stage}_max_act/"
# max_act_p = store_p + "test_max_act/" 



#endregion



#region Loads

macts= []

for ly in to_load:

	load_p = max_act_p + f"maxact_sparse_{ly}_{cp}.pkl"

	with open(load_p,"rb") as f:
		macts.append(pickle.load(f))

#endregion



#region Tokenizer to decode the tokens


#region* Model, for the tokenizer
seq_len = 512
model_name = "bert-base-uncased" #! THIS SHOULD BE UNCASED
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LanguageModel(model_name, device_map=device)
#endregion

tokenizer = model.tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.backend_tokenizer.enable_truncation(max_length=seq_len)
tokenizer.backend_tokenizer.enable_padding(length=seq_len, pad_id=tokenizer.pad_token_id, pad_token=tokenizer.pad_token)

#endregion  



#region Main


m = macts[0]

ks = get_subkeys(m)


#? To show topk of a specific feature f
# feat_show_topK(m,1,f,tokenizer,print=True)

mfs = [3478,4459,3754,3743,2953,4102,5239,5622,1379,3926,5527,3351,3919,]
ffs = [5221,389,4596,4307,830,3143,2038,5417,3834,4677,2059,2356,2746,]

m_tks = [feat_show_topK(m,1,feat,tokenizer,print=False) for feat in mfs]
f_tks = [feat_show_topK(m,1,feat,tokenizer,print=False) for feat in ffs]


print("STOP")
print("STOP")



#endregion








#region# Old idea for gender


# #?Checking which if any of these match
# toptok_grps = []


# f_set = {"woman", "women", "girl", "girls", "female", "lady", "ladies","mrs", "ms", "miss", "she", "her", "hers","herself",}

# m_set = {"man", "men", "boy", "boys", "male", "lad", "lads","mr","he", "him", "his","himself",}


# for i in range(6144):
# 	toptok_grps.append(set([tokenizer.decode(m["metadata"][(1,i,j)]["token_id"]) for j in range(20)]))

# m_check = [i for i in range(len(toptok_grps)) if m_set.intersection(toptok_grps[i]) != set()]
# f_check = [i for i in range(len(toptok_grps)) if f_set.intersection(toptok_grps[i]) != set()]

# both_check = [i for i in (f_check + m_check) if ((i in f_check) and (i in m_check))]
# m_only = [i for i in m_check if i not in both_check]

#endregion














print("END FILE")
