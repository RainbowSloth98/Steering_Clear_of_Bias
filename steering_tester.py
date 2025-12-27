from tqdm import tqdm
import os
import torch
import json
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn
import pickle
import pandas as pd
from collections import namedtuple
from collections import Counter
from functools import partial
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from more_itertools import windowed
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers.tokenization_utils_base import BatchEncoding

from dictionary_learning import AutoEncoder
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from dictionary_learning.trainers import StandardTrainer
from nnsight import LanguageModel
from nnsight.module import Module

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedModel,PretrainedConfig
from safetensors.torch import load_file


#region? PREPROCESSING DESCRIPTION
#* We started with the original dataframe, and used .apply() to tokenize all the rows (slow)
#* Following this, we used a combination of create_overlapping_chunks() and .explode to create a new ds
#* This one has 252503 entries in it, where many are below the max 512.
#endregion







#region Params


#region? model_type
model_type = "bert"
# model_type = "bge"
#endregion

#region? tok_type
tok_type = "pool" #? Admittedly, we'll probably only be using pool, since CLS doesn't classify
# tok_type = "cls"
#endregion

#region? f_c
f_c = 2 #Use only 2nd
# f_c = 3 #Use only 3rd
# f_c = 4 #Use second to last
# f_c = 5 #Use last
#endregion

#? Used for debugging stuff with CUDA
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#endregion



#region Defs

tok_map = {"cls":"CLS","pool":"POOL"}

if(model_type == "bert"):
	model_name = "bert-base-uncased" #! UNcased, not cased...
	act_dim = 768
elif (model_type == "bge"):
	model_name = "BAAI/bge-large-en-v1.5"
	act_dim = 1024


device = "cuda"
seq_len = 512
exp_factor = 8
dict_size = act_dim * exp_factor
# batch_size = 250 #? Should divide data_size
batch_size = 70
data_size = 22299 #* 361668 if it were all the old standard.
# batch_size = 46 #* Divides data_size
# data_size = 244214 #* Old data
step = 0
steps = data_size/batch_size #Number of examples
# steps = 1 #? For testing porposes, one batch is enough


#endregion



#region Paths



root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/steerer/"
load_p = root_p + "loads/"
save_p = root_p + "saves/"
store_p = "/media/strah/344A08F64A08B720/Work_related/store/"


tens_ds_p = load_p + f"{model_type}_tokds/gentpick{f_c}_tokds.pt"

# sae_path = load_p + "pre1_sae6_7000.pt" old
sae_path = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/mact/loads/BTK_SAE6_144000.pt"



labs_p = load_p + "cleargen_labs_26_9_25.pkl"


#classhead path
# classhead_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/classifier/saves/bert/POOL/model_v2/checkpoint-1734/"
classhead_p = root_p + "classhead/checkpoint-1650/"


mact_namer = "maxact_sparse_6_7000.pkl"
mact_p = store_p + mact_namer

#endregion



#region Classes


class ActDataset(Dataset):
	def __init__(self, input_ids, labels, pad_id=0):
		self.input_ids = input_ids
		self.labels = labels
		self.pad_id = pad_id

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, i):
		input_id = self.input_ids[i]
		
		# 1. Generate Attention Mask
		# Create a mask of 1s where the token is NOT padding, and 0s where it is.
		attention_mask = (input_id != self.pad_id).long()
		
		# 2. Generate Token Type IDs
		# For single sequence tasks, this is always a vector of zeros.
		token_type_ids = torch.zeros_like(input_id)

		return {
			"input_ids": input_id,
			"attention_mask": attention_mask,
			"token_type_ids": token_type_ids,
			"labels": self.labels[i]
		}


class ClassHead(nn.Module):
	def __init__(self, hidden_d):
		super().__init__()
		self.classifier = nn.Linear(hidden_d,1)


	def forward(self,input=None,labels=None):
		logits = self.classifier(input)
		loss = None

		if labels is not None:
			loss_fn = nn.BCEWithLogitsLoss()
			#Labels have to be float, since its expected by BCEWithLogitsLoss
			loss = loss_fn(logits.view(-1), labels.float())


		return {"logits":logits, "loss":loss}


class CustomModCon(PretrainedConfig):
	def __init__(self, hidden_size=768,**kwargs):
		super().__init__(**kwargs)
		self.hidden_size = hidden_size


class WrapClassModel(PreTrainedModel):

	def __init__(self,config):
		super().__init__(config)
		self.classifier = ClassHead(config.hidden_size)

	def forward(self,input=None,labels= None):

		return self.classifier(input=input,labels=labels)


#endregion



#region Class head


# Load the config manually from the JSON

with open(f"{classhead_p}config.json") as f:
	cfg_data = json.load(f)

# Create the model directly
model_cfg = CustomModCon(hidden_size=cfg_data["hidden_size"])
classhead = WrapClassModel(model_cfg)

# Load the weights directly
state_dict = load_file(f"{classhead_p}model.safetensors", device="cpu")
classhead.load_state_dict(state_dict)

classhead.to(device)
classhead.eval()




#endregion



#region Funcs



#Takes a recreated feature tensor, and idx of feat of interest, and shows distribution
def hist(tens: torch.Tensor, idx: int, show=False, ret=True):
	"""
	Creates a Plotly bar chart of SAE activations.
	
	Args:
		tens (torch.Tensor): Activation tensor of size [6144].
		idx (int): The index of the main feature to highlight.
	"""
	# Ensure tensor is on CPU and numpy-compatible
	data = tens.detach().cpu().numpy()
	
	# 1. Define Base Colors (Landscape)
	# specific hex for light grey ensures it fades into background
	colors = np.array(['#E2E2E2'] * len(data)) 
	
	# 2. Identify Top 5 Co-activations (excluding the target idx)
	# We set the target idx to -infinity temporarily so it doesn't get picked as a co-activation
	masked_data = data.copy()
	masked_data[idx] = -np.inf
	
	# Get indices of the top 5 highest values from the rest
	top_5_indices = np.argsort(masked_data)[-5:]
	
	# Highlight Co-activations (e.g., in Blue)
	colors[top_5_indices] = '#636EFA' 
	
	# 3. Highlight the Selected Target Feature (e.g., in Red)
	colors[idx] = '#EF553B'

	# Create the Bar Chart
	fig = go.Figure()
	
	fig.add_trace(go.Bar(
		x=np.arange(len(data)),
		y=data,
		marker_color=colors,
		# Remove borders on bars for cleaner look at high density
		marker_line_width=0, 
		hoverinfo='x+y',
		name='Activations'
	))

	# Layout for the "Dashboard" feel
	fig.update_layout(
		title=f"SAE Feature Activations (Target: {idx})",
		xaxis_title="Feature Index",
		yaxis_title="Activation Value",
		template="plotly_white",
		bargap=0, # Removes gaps to show the 'landscape' density better
		showlegend=False
	)

	if ((not show) and (not ret)):
		raise Exception("At least one of show or ret need to be true")

	if (show):
		fig.show()

	if ret:
		return fig


#Like show_hist, but only shows the non-zero entries.
def active_hist(tens: torch.Tensor, idx: int, show=False, ret=True):
	"""
	Displays only non-zero activations, compressed side-by-side.
	Preserves original indices in hover data.
	"""
	# 1. Filter Data: Keep only non-zero entries
	# We maintain the tensor on CPU/Numpy for easy indexing
	full_data = tens.detach().cpu().numpy()
	
	# Get indices where activation > 0
	# nonzero returns a tuple of arrays, we want the first one
	nz_indices = np.nonzero(full_data)[0] 
	nz_values = full_data[nz_indices]
	
	# If for some reason the target idx isn't in the non-zero list 
	# (e.g. looking at a dead neuron), we force include it or handle gracefully.
	# Here we assume idx is always active as per your "top entry" description.
	
	# 2. Setup Colors
	# Default color for all active features
	colors = np.array(['#B4B4B4'] * len(nz_indices)) # Slightly darker grey for visibility
	
	# Find where the target 'idx' is located in our new compressed 'nz_indices' array
	# np.where returns a tuple, we take the first element (array of indices) then the first item
	target_loc_array = np.where(nz_indices == idx)[0]
	
	if len(target_loc_array) > 0:
		target_pos = target_loc_array[0]
		colors[target_pos] = '#EF553B' # Red for target
		
		# 3. Find Top 5 Co-activations
		# Create a masked copy of values to find top 5 (excluding the target)
		masked_values = nz_values.copy()
		masked_values[target_pos] = -np.inf
		
		# Get indices of top 5 values in the *compressed* array
		if len(masked_values) > 1: # Only look for co-activations if there are other features
			# argsort sorts ascending, so we take the last 5
			top_5_local_indices = np.argsort(masked_values)[-5:]
			colors[top_5_local_indices] = '#636EFA' # Blue for co-activations
	
	# 4. Create Plot
	fig = go.Figure()
	
	fig.add_trace(go.Bar(
		# X is just 0, 1, 2... N (compressed sequence)
		x=np.arange(len(nz_indices)), 
		y=nz_values,
		marker_color=colors,
		marker_line_width=0,
		
		# 5. Inject the ORIGINAL Feature Indices into the hover data
		customdata=nz_indices,
		hovertemplate=(
			"<b>Feature Index: %{customdata}</b><br>" +
			"Activation: %{y:.4f}<br>" +
			"<extra></extra>" # Removes the secondary box named 'trace 0'
		)
	))

	fig.update_layout(
		title=f"Active Features for  (Target: {idx}) - Showing {len(nz_indices)}/{len(full_data)}",
		xaxis_title="Active Features (Ordered by Index)",
		yaxis_title="Activation Value",
		template="plotly_white",
		bargap=0.1, # Slight gap to distinguish distinct bars
		showlegend=False
	)


	if ((not show) and (not ret)):
		raise Exception("At least one of show or ret need to be true")

	if (show):
		fig.show()

	if ret:
		return fig


#Generator for creating overlapping chunks
def create_overlapping_chunks(tokens, chunk_size=512, overlap=100):
	"""Yields successive overlapping chunks from a list of tokens."""
	if not tokens:
		return
	
	# The step size is the chunk size minus the overlap
	step = chunk_size - overlap
	for i in range(0, len(tokens), step):
		yield tokens[i:i + chunk_size]



def _printer(pre,post):
	print(pre)
	print("#"*50)
	print(post)



#region* monkey patching torch.load

og_torch_load = torch.load

def ntorch_load(*args, **kwargs):
	kwargs['weights_only'] = False
	return og_torch_load(*args, **kwargs)

torch.load = ntorch_load #?its good practice to restore the function when done

#endregion



def single_mactensor_recreate(indies,vals):
	
	cand = torch.zeros([6144]).to(torch.float32)

	#Broadcasting with torch makes this fast
	cand[indies] = vals

	return cand



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



#region Model

model = LanguageModel(model_name,device_map=device)
model.eval() #No changing weights.

if (model_type == "bert"):
	submodule_ref6 = eval("model.bert.encoder.layer[6]") #! The eval allows NNSight to work.
	submodule_ref7 = eval("model.bert.encoder.layer[7]")
	submodule_ref_last = eval("model.bert.encoder.layer[11]")
elif (model_type == "bge"):
	pass


#endregion



#region Tokenizer

tok = model.tokenizer
tok.add_special_tokens({'pad_token': '[PAD]'})
tok.backend_tokenizer.enable_truncation(max_length=seq_len)
tok.backend_tokenizer.enable_padding(length=seq_len, pad_id=tok.pad_token_id,pad_token=tok.pad_token)

#endregion



#region SAE

# with open(sae_path,"rb") as f:
# 	ae_state_dict = torch.load(f)

# 	ae6 = AutoEncoder(activation_dim = act_dim, dict_size=dict_size)
# 	ae6.load_state_dict(ae_state_dict)
# 	ae6.to(device)

ae6 = BatchTopKSAE.from_pretrained(sae_path,device=device)

#! For steering to work, we need to register our SAE with the nnsight framework
model.ae6 = ae6


#endregion



#region Loading Data and Labs


with open(labs_p,"rb") as f:
	labs = pickle.load(f)

#?Loading a TensorDataset version of our data
with open(tens_ds_p,"rb") as f:
	tens_ds = torch.load(f)

#ActDataset
ad_inputs = tens_ds.tensors[0]
ad_labels = torch.tensor(labs, dtype=torch.long)


train_ds = ActDataset(ad_inputs, ad_labels, pad_id=tok.pad_token_type_id)


#endregion Loading Data and Labs



#region Sampler, ensuring balanced batches

#? Avoiding type err masked as "classes should include all valid lables" err
y_np = labs.to(torch.int).numpy()

#? Automatically compute weights
class_weights = compute_class_weight(
	class_weight='balanced', 
	classes=np.unique(y_np), 
	y=y_np
)


# # Map class weight to individual sample
# weights_tensor = torch.tensor(class_weights, dtype=torch.float)

weight_map = dict(zip(np.unique(y_np), class_weights))


# # This creates a tensor where every sample has the weight of its class
# sample_weights = weights_tensor[lab_tensor]
samples_weight = torch.tensor([weight_map[x] for x in y_np], dtype=torch.float)



# 3. Create the sampler
sampler = WeightedRandomSampler(
	weights=samples_weight,
	num_samples=len(samples_weight),
	replacement=True
)


#endregion Sampler, ensuring balanced batches



#region DataLoader def


loader = torch.utils.data.DataLoader(
	train_ds,
	batch_size=batch_size,
	sampler=sampler,
	shuffle=False,
	num_workers=4,
	prefetch_factor=8,
	persistent_workers=True,
)


diter = iter(loader)


#endregion DataLoader def



#region# Mact load

# #Max acts
# with open(mact_p,"rb") as f:
# 	mact= pickle.load(f)

#endregion#



#region# Diff_Vecs (Where applicable)

#########################################
#region# T1: Difference vectors

# diff_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/steerer/steering_vecs/"
# ns = ["guy_minus_girl.pt","guy_minus_boy.pt","boy_minus_girl.pt"]
# diffs = []
# dif_co = 1 

#region# T1: Loading diffs

# for n in ns:
# 	dn_p = diff_p + n
# 	with open(dn_p,"rb") as f:
# 		diffs.append(torch.load(f))

#endregion


#endregion
#########################################
#region# T2: Load avg difference vec

# avg_diff_p = diff_p + "avg_masc-fem_vec_v1.pt"

# with open(avg_diff_p,"rb") as f:
# 	avg_diff_v1 = torch.load(f)

# #* Defining the scaler.
# avg_diff_scaler = 3


#endregion
#########################################
#region# T3: Global Difference (same as T2)

#region*# Naive ADVec, No good 

# avg_diff_p = diff_p + "avg_masc-fem_vec_v1.pt"

# with open(avg_diff_p,"rb") as f:
# 	avg_diff_vec = torch.load(f)

# #Defining the scaler.
# avg_diff_scaler = 3


#endregion	

#region*# Denoise by taking only those above 0.2 and below -0.2

# interdexes = [
# 	2346,2353,898,4866,3787,381, #Masc main. Interps: [him],[he],[he],[he+would/could/can etc.], [he,his], [his/my]
# 	3719, #Masc side. Interps: [PAD]
# 	4081,4816,2687,4868, #Fem main. Interps: [Elizabeth,Catherine,widow,wife etc.], [general her], [rand fems], [her/she after prepositions]
# 	5284,408,3714,2063,# Fem side. Interps: [mother etc.], [|masc| or her/wife],[Family terms] ,[She/Her, first cap]
# 	]

# intervals = [
# 	0.5340,0.4885,0.4829,0.4694,0.2582,0.28,#Masc main
# 	0.2135, #Masc side
# 	1.3966,1.0132,0.5708,0.5120, #Fem main
# 	0.7836,0.6872,0.2297,0.2077, #Fem side
# 	]

# #? T3.1: Modified from the actual average value, specifically in positions 7(OG*0.55),8(OG*1.25) and 11(OG*0.5)
# #? T3.2.1: Above helped, but still needs to be better; Try relatively equalizing 7,8,9 (set 0.5). Kept 11 from 3.1 
# #? T3.2.2: Found from princess embedding that 2687 should probs be higher than 4816
# intervals = [
# 	0.5340,0.4885,0.4829,0.4694,0.2582,0.28,#Masc main
# 	0.2135, #Masc side
# 	0.5681,0.5665,0.7708,0.5120, #Fem main
# 	0.3918,0.6872,0.2297,0.2077, #Fem side
# 	]

# fig_test = show_act(intervals,"T3.2: Equalize 7,8,9")
# # fig_test.show()


# interall = list(zip(interdexes,intervals))

# intervec = torch.zeros([6144])
# for i,v in interall:
# 	intervec[i] = v

#? Testing here
# intervec_scal = 1 #* Does an ok but incomplete job; Some corruptions
# intervec_scal = 1.5 #* Does a better job, but proporitonally more corruption
# intervec_scal = 2 #* As expected more effects, more corruption; Daughter overrepresented
# intervec_scal = 2.5 #* More effects, Now also seeing a change from man to husband as well.
# intervec_scal = 3 #* Expecting chaos. 





#endregion

#endregion
#########################################


#endregion#



#region Input Select


#region# Other inputs

#region TODO Real input

#? No more doubling up - included a sampler
# batch = next(diter)
# token_type_ids = torch.zeros(batch[0].size())
# #Reconstructing the BatchEncoding object's dictionary manually.
# inputs = {"input_ids":batch[0].type(torch.int64).to(device),"token_type_ids":token_type_ids,"attention_mask":batch[1]}

#endregion
#########################################
#region*# T1: Toy inputs for guygirl testing steering

# s1t1 = "Female people are known as women and girls. Male people are known as boys and men. There are also others."
# s2t1 = "Andrew is the most popular guy in all of the school; he's the captain of the football team and a natural leader who everyone looks up to."
# s3t1 = "Robert is the most popular boy in all of the school; his easygoing charm and quick wit mean he's always the center of attention in any social setting."
# s4t1 = "Stacy is the most popular girl in all of the school; her impeccable sense of style and involvement in organizing social events make her admired by everyone."
# test1 = [s1t1,s2t1,s3t1,s4t1]

#endregion*# T1: Toy inputs for guygirl testing steering
#########################################
#region* T2: Contrasting pairs for gender


#region** T2 Examples

m0t2 = "Joseph is male, and he knows it."
f0t2 = "Kate is female, and she knows it."

m1t2 = "Danny is a beautiful guy, or at least he thinks so."
f1t2 = "Ella is a beautiful gal, or at least she thinks so."

m2t2 = "Ollie is a lovely boy, he always behaves nicely."
f2t2 = "Jenny is a lovely girl, she always behaves nicely."

m3t2 = "James is a real man, he does many things to show it."
f3t2 = "Juliet is a real woman, she does many things to show it."

m4t2 = "The passport says the traveller is male."
f4t2 = "The passport says the traveller is female."

m5t2 = "He went to the shops yesterday."
f5t2 = "She went to the shops yesterday."

m6t2 = "The wolf was hungry, so it ate him."
f6t2 = "The wolf was hungry, so it ate her."

m7t2 = "The teacher was grading the class, and asked Adam to see his work."
f7t2 = "The teacher was grading the class, and asked Betty to see her work."

m8t2 = "You must act proper in order to be considered a gentleman."
f8t2 = "You must act proper in order to be considered a lady."

#endregion**# T2 Examples


#region** T2 Formatting Examples

# test2 = [m0t2,f0t2,m1t2,f1t2,m2t2,f2t2,m3t2,f3t2,m4t2,f4t2,m5t2,f5t2,m6t2,f6t2,m7t2,f7t2,m9t2,f9t2]
#? Setting this for ease of testing before and after
test2 = [
	m0t2,f0t2,f0t2,
	m1t2,f1t2,f1t2,
	m2t2,f2t2,f2t2,
	m3t2,f3t2,f3t2,
	m4t2,f4t2,f4t2,
	m5t2,f5t2,f5t2,
	m6t2,f6t2,f6t2,
	m7t2,f7t2,f7t2,
	m8t2,f8t2,f8t2]

#endregion** T2 Formatting Examples



#endregion* T2: Contrasting pairs for gender
#########################################
#region*# T3: Global Toy Inputs

# test3 = ["The king watched his son, the young prince. The queen's gaze, however, was on her daughter, the princess. Nearby, a lord bowed to a lady; the man's gesture was met by the woman's polite nod. The conflict's hero had a famously masculine confidence, contrasting with the heroine's more subtle and feminine style of leadership. A young boy ran to his father, while a nearby girl held her mother's hand. In the corner, an actor rehearsed lines with an actress; the sister's delivery was flawless, but her brother forgot his cue. The actress's husband smiled at the error, though the actor's wife looked on with a critical eye.", "The king watched his son, the young princess."]

#endregion*# T3: Global Toy Inputs
#########################################
#region*# Test input preproceesing

sentences = test2 #! Change here for different tests (test1/2/3)
encs = torch.Tensor([tok.encode(s)+[0]*(512-len(tok.encode(s))) for s in sentences]).to(torch.int64)
attens = (encs != 0).to(torch.long)
ttids = torch.zeros(encs.size()).to(torch.long)
trace_inputs = {"input_ids":encs,"token_type_ids":ttids,"attention_mask":attens}

#endregion*# Test input preproceesing
#########################################
#endregion# Test inputs


#region TODO T5: Testing batches, no steering.

# bat = next(diter)
# # batch = next(diter)

# #? Doing two comparisons, one going raw, the other with recon.
# nbat = {}
# for i in bat.keys():
# 	nbat['input_ids'] = torch.cat([bat['input_ids'],bat['input_ids']])
# 	nbat['labels'] = torch.cat([bat['labels'],bat['labels']])
# 	nbat['attention_mask'] = torch.cat([bat['attention_mask'],bat['attention_mask']])
# 	nbat['token_type_ids'] = torch.cat([bat['token_type_ids'],bat['token_type_ids']])


#? Fron 18.11.25, we're doing this automatically in ActDataset.
# inputs = {"input_ids":nbat[0].type(torch.int64).to(device),"token_type_ids":token_type_ids,"attention_mask":nbat[1]}

#endregion TODO T5: Testing batches/recons - no steering.


#endregion Input Select



#region Steering

#? Make sure to do a .clone, otherwise the .saves get executed at the same time at the end.

#region* Fixing "inputs", no labels!
#? This doesn't work because 
# inputs = nbat
# trace_inputs = {k: v for k, v in nbat.items() if k != "labels"}
# batch_labels = nbat["labels"].to(device)

#endregion* Fixing "inputs", no labels!



with torch.inference_mode():
	with model.trace(trace_inputs) as tracer:
	
		acts = submodule_ref6.nns_output[0]

		to_sae = acts
		# no_sae = acts[:batch_size] #? Part of no steer / recon test T6
		# to_sae = acts[batch_size:]
		
		#? Into the steerer
		sae_hidden = model.ae6.encode(to_sae)

		sae_hidsav = sae_hidden.clone().to("cpu").save()

		#region# Steer testing sae_hidden

		# region# Real Steering
		
		# #? Modified to only apply to the first example
		# #Find the OG magnitude
		# hidden_mag = torch.linalg.norm(sae_hidden[:,1:,],dim=-1,keepdim=True)
		# #Modify the values
		# new_vec = (sae_hidden[:,1:,:,] + intervec_scal*intervec).relu().to(torch.float32)
		# #Create a unit vector in the new dimension
		# unit_vec = F.normalize(new_vec,p=2,dim=-1)
		# #Update the hidden state
		# sae_hidden[:,1:,] = unit_vec * hidden_mag

		#endregion# Real Steering

		#########################################
		#region*# T1: Adding guy-girl in theory makes more masculine

		# no_c = sae_hidden.clone().to("cpu").save()
		# sae_hidden[2,6,:] += diffs[0]*dif_co #? boy
		# boy_c = sae_hidden.clone().to("cpu").save()
		# sae_hidden[3,6,:] += diffs[0]*dif_co #? girl
		# boygirl_c = sae_hidden.clone().to("cpu").save()

		#endregion
		#########################################
		#region*# T2: Steer and save 
		
		
		
		# gender_diffs = []
		
		# m_exs = [
		# 	sae_hidden[0,0,:].to("cpu").save(),
		# 	sae_hidden[3,0,:].to("cpu").save(),
		# 	sae_hidden[6,0,:].to("cpu").save(),
		# 	sae_hidden[9,0,:].to("cpu").save(),
		# 	sae_hidden[12,0,:].to("cpu").save(),
		# 	sae_hidden[15,0,:].to("cpu").save(),
		# 	sae_hidden[18,0,:].to("cpu").save(),
		# 	sae_hidden[21,0,:].to("cpu").save(),
		# 	sae_hidden[24,0,:].to("cpu").save(),

		# ]
		
		# f_exs = [
		# 	sae_hidden[1,0,:].to("cpu").save(),
		# 	sae_hidden[4,0,:].to("cpu").save(),
		# 	sae_hidden[7,0,:].to("cpu").save(),
		# 	sae_hidden[10,0,:].to("cpu").save(),
		# 	sae_hidden[13,0,:].to("cpu").save(),
		# 	sae_hidden[16,0,:].to("cpu").save(),
		# 	sae_hidden[19,0,:].to("cpu").save(),
		# 	sae_hidden[22,0,:].to("cpu").save(),
		# 	sae_hidden[25,0,:].to("cpu").save(),
		# ]

		# gender_diffs.append((sae_hidden[0,0,:] - sae_hidden[1,0,:]).to("cpu").save())
		# gender_diffs.append((sae_hidden[3,0,:] - sae_hidden[4,0,:]).to("cpu").save())
		# gender_diffs.append((sae_hidden[6,0,:] - sae_hidden[7,0,:]).to("cpu").save())
		# gender_diffs.append((sae_hidden[9,0,:] - sae_hidden[10,0,:]).to("cpu").save())
		# gender_diffs.append((sae_hidden[12,0,:] - sae_hidden[13,0,:]).to("cpu").save())
		# gender_diffs.append((sae_hidden[15,0,:] - sae_hidden[16,0,:]).to("cpu").save())
		# gender_diffs.append((sae_hidden[18,0,:] - sae_hidden[19,0,:]).to("cpu").save())
		# gender_diffs.append((sae_hidden[21,0,:] - sae_hidden[22,0,:]).to("cpu").save())
		# gender_diffs.append((sae_hidden[24,0,:] - sae_hidden[25,0,:]).to("cpu").save())


		# sae_hidden[2,3,:] = (sae_hidden[2,3,:] + avg_diff_scaler*avg_diff_v1).relu()/2
		# sae_hidden[5,5,:] = (sae_hidden[5,5,:] + avg_diff_scaler*avg_diff_v1).relu()/2
		# sae_hidden[8,5,:] = (sae_hidden[8,5,:] + avg_diff_scaler*avg_diff_v1).relu()/2
		# sae_hidden[11,5,:] = (sae_hidden[11,5,:] + avg_diff_scaler*avg_diff_v1).relu()/2
		# sae_hidden[14,8,:] = (sae_hidden[14,8,:] + avg_diff_scaler*avg_diff_v1).relu()/2
		# sae_hidden[17,1,:] = (sae_hidden[17,1,:] + avg_diff_scaler*avg_diff_v1).relu()/2
		# sae_hidden[20,9,:] = (sae_hidden[20,9,:] + avg_diff_scaler*avg_diff_v1).relu()/2
		# sae_hidden[23,14,:] = (sae_hidden[23,14,:] + avg_diff_scaler*avg_diff_v1).relu()/2
		# sae_hidden[26,11,:] = (sae_hidden[26,11,:] + avg_diff_scaler*avg_diff_v1).relu()/2

		

		

		#endregion
		#########################################
		#region*# T3: Global steering 
		
		#region*# T3.1: Full avg_diff does not work; Output very corrupted

		# # sae_hidden[:,:,] += avg_diff_scaler * avg_diff_vec

		#endregion

		#region*# T3.2: Change the distribution, keep the mag


		# # #? Modified to only apply to the first example
		# #Find the OG magnitude
		# hidden_mag = torch.linalg.norm(sae_hidden[0,:,])
		# #Modify the values
		# new_vec = (sae_hidden[0,:,] + intervec_scal*intervec).relu().to(torch.float32)
		# #Create a unit vector in the new dimension
		# unit_vec = F.normalize(new_vec,p=2,dim=-1)
		# #Update the hidden state
		# sae_hidden[0,:,] = unit_vec * hidden_mag



		#endregion

		#endregion
		##########################################

		#endregion

		#? Out of the steerer
		recons = model.ae6.decode(sae_hidden)


		# outs = torch.cat([no_sae,recons]) #? Part of no steer / recon test T6

		outs = recons

		submodule_ref7.input = outs


		#* Save after steering is done
		finl = model.output.save()
		l11 = submodule_ref_last.nns_output.save()


#endregion Steering



#region Main Analysis


#region* Post-Processing


out11 = l11[0] #? Unprocessed outputs


if (tok_type == "cls"):
	fin_mean = finl[0].to(device,non_blocking=True)
elif (tok_type == "pool"):
	fin_mean = torch.mean(l11[0],dim=1) #.to(device,non_blocking=True)

#endregion* Post-Processing



#region curr Analysing BTK


with torch.inference_mode():

	#region*# Testing pooled! (Works)
	
	# control  = fin_mean[:batch_size]
	# reconed = fin_mean[batch_size:]

	# control_classed = classhead(control)
	# reconed_classed = classhead(reconed)

	# A = control_classed["logits"]
	# B = reconed_classed["logits"]


	#region** Helper funcs
	
	#Aux function to help with printing
	# def p(i="n"):

	# 	if(i == "n"): #Default print all
	# 		print(A)
	# 		print("#"*100)
	# 		print(B)
	# 		print("#"*100)
		
	# 	else: #Print individual, put in loop to compare directly
	# 		print(A[i])
	# 		print("#"*10)
	# 		print(B[i])
	# 		print("#"*10)

	# 	pass


	def ret_tens_vis(tensor_data, name):
		"""
		Visualizes a torch.Tensor of shape [70, 1] as a Plotly bar chart.
		
		Args:
			tensor_data (torch.Tensor): Input tensor of shape [70, 1]
		"""
		# 1. Detach from graph (if necessary), move to CPU, convert to numpy, and flatten
		# This ensures the code works for tensors on GPU or with gradients
		data = tensor_data.detach().cpu().numpy().flatten()
		
		# 2. Create the bar chart
		fig = px.bar(
			x=range(len(data)), 
			y=data, 
			labels={'x': 'Index', 'y': 'Value'},
			title=f"Visualization for {name}"
		)
		
		# 3. Show the interactive plot
		# fig.show()
		return fig

	#endregion* Helper funcs


	# print("STOP")
	# print("STOP")

	#endregion*# Testing pooled! (Works)


	#region*# Testing unpooled. (Works)
	

	#region** Helper function

	def full_act_compare(control, reconstruction):

		N, D = control.shape
		
		# --- 1. Compute Metrics ---
		# We compute cosine similarity for every pair [i]
		cosine_sims = []
		
		print("Computing similarities...")
		# using tqdm as requested for loops
		for i in tqdm(range(N), desc="Processing Batch"):
			c_vec = control[i].unsqueeze(0)
			r_vec = reconstruction[i].unsqueeze(0)
			sim = F.cosine_similarity(c_vec, r_vec)
			cosine_sims.append(sim.item())
			
		cosine_sims = np.array(cosine_sims)
		
		# --- 2. Compute Similarity Matrix (Subset) ---
		# For the heatmap, we use a subset to keep the webGL rendering smooth.
		# We visualize the first 50 samples to check alignment.
		subset_size = min(50, N)
		
		# Normalize vectors to get cosine similarity via dot product
		c_norm = F.normalize(control[:subset_size], p=2, dim=1)
		r_norm = F.normalize(reconstruction[:subset_size], p=2, dim=1)
		
		# Matrix multiplication: [Subset, D] @ [D, Subset] -> [Subset, Subset]
		# Range is -1 to 1
		similarity_matrix = torch.mm(c_norm, r_norm.t()).detach().cpu().numpy()

		# --- 3. Build Plotly Figure ---
		fig = make_subplots(
			rows=1, cols=2,
			column_widths=[0.4, 0.6],
			subplot_titles=(
				f"Distribution of Similarity<br>(Mean: {cosine_sims.mean():.3f})", 
				f"Alignment Matrix (First {subset_size} samples)"
			)
		)

		# Plot A: Histogram
		fig.add_trace(
			go.Histogram(
				x=cosine_sims,
				nbinsx=40,
				marker_color='#636EFA',
				name='Sim Distribution',
				hovertemplate='Similarity: %{x:.2f}<br>Count: %{y}<extra></extra>'
			),
			row=1, col=1
		)

		# Add a vertical line for the mean
		fig.add_vline(
			x=cosine_sims.mean(), 
			line_width=3, 
			line_dash="dash", 
			line_color="red", 
			annotation_text="Mean", 
			row=1, col=1
		)

		# Plot B: Heatmap
		# We want the diagonal to be bright yellow (1.0) and off-diagonal to be dark (0.0 or negative)
		fig.add_trace(
			go.Heatmap(
				z=similarity_matrix,
				x=list(range(subset_size)),
				y=list(range(subset_size)),
				colorscale='Viridis',
				zmin=0, zmax=1,  # Clamping visual range for better contrast
				name='Alignment',
				hovertemplate='Control Idx: %{y}<br>Recon Idx: %{x}<br>Sim: %{z:.3f}<extra></extra>'
			),
			row=1, col=2
		)

		# Layout Polish
		fig.update_layout(
			title_text="Control vs. Reconstruction Analysis",
			height=600,
			showlegend=False,
			template="plotly_white"
		)
		
		# Label axes
		fig.update_xaxes(title_text="Cosine Similarity", row=1, col=1)
		fig.update_yaxes(title_text="Count", row=1, col=1)
		fig.update_xaxes(title_text="Reconstruction Index", row=1, col=2)
		fig.update_yaxes(title_text="Control Index", row=1, col=2)

		# fig.show()
		
		return fig

	#endregion** Helper function

	# cont = out11[:batch_size]
	# recon = out11[batch_size:]

	# #Create a dictionary for comparing the figures
	# w = {f"{i}": full_act_compare(cont[i],recon[i]) for i in range(batch_size)}



	# print("STOP")
	# print("STOP")
	
	#endregion*# Testing unpooled.


	#region* Finding gender related feats
	


	#Mask to remove the duplicate sentences from the input; Reducing 27 -> 18
	m1  = [
		True,True,False,
		True,True,False,
		True,True,False,
		True,True,False,
		True,True,False,
		True,True,False,
		True,True,False,
		True,True,False,
		True,True,False,]

	#Mask for getting the specific gendered words
	m2 = [
		3,3, # male,female
		5,5, # Danny, Ella
		5,5, # Ollie, Jenny
		5,5, # James, Juliet
		7,7, # male , female (last word)
		1,1, # He, She
		9,9, # him, her
		10,10, # his, her
		11,11, # gentleman, lady
		]

	t = sae_hidsav[m1]
	gends = t[:,m2,:]

	#sae_hidsav
	it = list(zip(range(18),m2))


	#region** Helper F
	
	def get_feat_from_gends(feat, gends):

		vals = []
		it = list(zip(range(18),m2))
		
		for i,w in it:
			vals.append(gends[i][w][feat])

		return vals
	
	#endregion** Helper F


	all_figs = []
	topks = []

	for i,w in it:
		all_figs.append(hist(gends[i][w],0))
		topks.append(gends[i][w].topk(15))

	c = Counter()

	for tk in topks:
		c.update(tk[1].tolist())
	


	print("STOP")
	print("STOP")

	#endregion* Finding gender related feats




#endregion curr Analysing BTK



#region# Testing Analysis
#########################################
#region# T1: Changed boy to guy. Girl did not work.

# Create more diverse sets of sentences to get a more accurate steering vector

# 	#* Getting the acts
# #6144 features defining 
# guy_act = no_c[1,6,:]

# #Pre intervention
# pre_girl_act = no_c[3,6,:]
# pre_boy_act = no_c[2,6,:]

# #Post intervention
# post_girl_act = boygirl_c[3,6,:]
# post_boy_act = boygirl_c[2,6,:]

# #* Show via figs
# guy_fig = show_act(guy_act,"'guy' features")
# pre_girl_fig = show_act(pre_girl_act,"OG 'girl' features")
# pre_boy_fig = show_act(pre_boy_act,"OG 'boy' features")
# post_girl_fig = show_act(post_girl_act,"'girl' after intervention")
# post_boy_fig = show_act(post_boy_act,"'boy' after intervention")


# #* Decode to see if it works

# OG_toks = inputs["input_ids"]
# wordsin = list(map(tok.decode,OG_toks))

# wordsout = []
# wordsout.append(tok.decode(torch.argmax(finout[1],dim=-1)))
# wordsout.append(tok.decode(torch.argmax(finout[2],dim=-1)))
# wordsout.append(tok.decode(torch.argmax(finout[3],dim=-1)))

#endregion# T1: Changed boy to guy. Girl did not work.
#########################################
#region# T2: Post steering analysis 


#region*# Creating avg_diff vector
#? Found that it was basically exactly the same (bar fp_num nonsense)

# avg_diff_v2 = torch.zeros(6144)
# for ex in gender_diffs:
# 	avg_diff_v2 += ex
# avg_diff_v2 = avg_diff_v2 / len(gender_diffs)


# #Compare the two, are they the same?
# avg_v1_fig = show_act(avg_diff_v1,"Average Differences vector version 1")
# avg_v2_fig = show_act(avg_diff_v2,"Average Differences vector version 2")


# print("STOP")
# print("STOP")

#endregion*# Creating avg_diff vector


#region*# Other analysis


# OG_toks = inputs["input_ids"]
# wordsin = list(map(tok.decode,OG_toks))




# ind2 = [3,3,3,
# 		5,5,5,
# 		5,5,5,
# 		5,5,5,
# 		7,7,7,
# 		1,1,1,
# 		8,8,8,
# 		12,12,12,
# 		11,11,11,]

# ind1 = list(range(len(ind2)))

# main_fig_titles = ["Male",
# 				"OG_female",
# 				"New_female",
# 				"Guy",
# 				"OG_gal", 
# 				"New_gal", 
# 				"Boy",
# 				"OG_girl",
# 				"New_girl", 
# 				"Man",
# 				"OG_woman", 
# 				"New_woman", 
# 				"Male",
# 				"OG_female, p2", 
# 				"New_female, p2", 
# 				"He",
# 				"OG_She", 
# 				"New_She", 
# 				"Him",
# 				"OG_her", 
# 				"New_her", 
# 				"His",
# 				"OG_hers", 
# 				"New_hers", 
# 				"Gentleman",
# 				"OG_lady",
# 				"New_lady",
# 				]

# figs = []
# for i in range(len(ind2)):
# 	figs.append(show_act(post_hid_save[ind1[i],ind2[i],:].to("cpu"),main_fig_titles[i]))

# #Show!
# for f in figs:
# 	f.show()

#? Use to check predictions.
# out = tok.decode(torch.argmax(finout[1],dim=-1))


#endregion*# Other analysis	



#endregion# T2: Post steering analysis 
#########################################
#region# T3: Global Testing



# OG_toks = inputs["input_ids"][0] #Only one example
# wins = tok.decode(OG_toks)


# wouts = tok.decode(torch.argmax(finout[0],dim=-1))






#region*# If we ever need the old avg vec

# avg_diff_p = diff_p + "avg_masc-fem_vec_v1.pt"

# with open(avg_diff_p,"rb") as f:
# 	avg_diff_vec = torch.load(f)

# adv_fig = show_act(avg_diff_vec,"Average Difference Vector")

#endregion*# If we ever need the old avg vec

#endregion# T3: Global Testing
#########################################
#region# T4: CLS tests

# f_exs = torch.stack(f_exs)
# m_exs = torch.stack(m_exs)

# mmasks = [ m_exs[i] < 0.65 for i in range(len(m_exs))]
# mmasks = torch.stack(mmasks)
# fmasks = [ f_exs[i] < 0.65 for i in range(len(f_exs))]
# fmasks = torch.stack(fmasks)

# m_trimmed = torch.where(mmasks,m_exs,torch.zeros([6144]))
# f_trimmed = torch.where(fmasks,f_exs,torch.zeros([6144]))


# mf = (m_trimmed*100) - (f_trimmed*100)


# CLS_figs = [show_act(mf[i],f"Mtrimm - Ftrimm * 100 num {i}") for i in range(mf.size()[0])]

#endregion
#endregion# Testing Analysis



#endregion Main Analysis




















print("FILE END")
