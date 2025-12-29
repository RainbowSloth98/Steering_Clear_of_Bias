import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import gc

from dictionary_learning import AutoEncoder
from nnsight import LanguageModel
from nnsight.module import Module


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
from transformers import PreTrainedModel,PretrainedConfig
from safetensors.torch import load_file



#region# Balancing dataset code


# nzs = (full_labs == 1).nonzero(as_tuple=True)[0]

# nzs = nzs[torch.randperm(sample_per_class)]

# zs = (full_labs == 0).nonzero(as_tuple=True)
# #? Create random permutation of idxs for negatives, and only take the amount to equal positives
# zs_r = zs[0][torch.randperm(len(zs[0]))][:len(nzs)]

# idxs = torch.cat([nzs,zs_r])
# idxs_r = idxs[torch.randperm(len(idxs))]


# avg_act_p = loads_p + "avged_acts/"
# act_list = []
# for i in idxs_r.tolist():
# 	act_list.append(torch.load(avg_act_p+f"averaged_doc{i}.pt"))


#endregion# Balancing dataset code




#region? Info

#? Q: Does the INLP transform actually properly work?

#endregion? Info




#region Params


#region? frag_choice

#region? Choices
#? 1 = First chunk only
#? 2 = Second chunk only
#? 3 = Third chunk only; Where only 2 exist, take the latter.
#? 4 = prefinal/penultimate chunk only
#? 5 = final/ultimate chunk only
#endregion
frag_choice = 2

#endregion? frag_choice



#region? tok_type
tok_type = "pool" 
# tok_type = "cls"
tok_map = {"cls":"CLS","pool":"POOL"}
#endregion



#region? model_type
model_type = "bert"
# model_type = "bge"

if(model_type == "bert"):
    h_dim = 768 #for bert
elif(model_type == "bge"):
    h_dim = 1024 #for bge

#endregion? model_type



#region? act_select

act_select = "middle"
# act_select = "end"

if (act_select == "middle"):
	if (model_type == "bert"):
		act_type = 6
	elif(model_type == "bge"):
		act_type = 12
		
elif (act_select == "end"):
	if (model_type == "bert"):
		act_type = 11
	elif(model_type == "bge"):
		act_type = 23


#endregion? act_type



#endregion Params



#region Predefs
#? Hard coding in BERT, pooling and f_c=2

model_name = "bert-base-uncased" 
act_dim = 768
device = "cuda"
seq_len = 512
exp_factor = 8
dict_size = act_dim * exp_factor
# batch_size = 250 
batch_size = 10 #TODO For testing purposes only
data_size = 22299 #* 361668 if it were all the old standard.
step = 0
steps = data_size/batch_size #Number of examples
# steps = 1 #? For testing porposes, one batch is enough


#endregion Predefs



#region Classes


class ActDataset(Dataset):

	def __init__(self, actensors, label):
		# super().__init__()
		self.actensors = actensors
		self.label = label
	

	def __len__(self):
		return len(self.label)


	def __getitem__(self, i):
		return {"input":self.actensors[i],"label":self.label[i]}


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



#endregion Classes



#region Paths

root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/pytest/"
steerer_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/steerer/"
inlp_rp = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/INLP/saves/"

inlp_trained_p = inlp_rp + "first_step.pkl"
inlp_calced_p = inlp_rp + "fin_step_t.pt"


saves_p = root_p + "saves/"
loads_p = root_p + "loads/"

gen_class_p = loads_p + f"l{act_type}_checkpoint-1263/"


steerloads_p = steerer_p + "loads/"

task_labs_p = steerloads_p + "cleargen_labs_26_9_25.pkl"
mf_labs_p = steerloads_p + "real_malefem_inlp_labs_7_11_25.pt"#? Taking the correct lables this time.
tens_ds_p = steerloads_p + f"bert_tokds/gentpick2_tokds.pt"

#endregion Paths



#region Model Load

model = LanguageModel(model_name,device_map=device)
model.eval() #No changing weights.

submodule_ref6 = eval("model.bert.encoder.layer[6]") 
submodule_ref7 = eval("model.bert.encoder.layer[7]")
submodule_ref_11 = eval("model.bert.encoder.layer[11]")


#endregion Model Load



#region Tokeniser Load

tok = model.tokenizer
tok.add_special_tokens({'pad_token': '[PAD]'})
tok.backend_tokenizer.enable_truncation(max_length=seq_len)
tok.backend_tokenizer.enable_padding(length=seq_len, pad_id=tok.pad_token_id,pad_token=tok.pad_token)


#endregion Tokeniser Load



#region Gender Classifier


with open(f"{gen_class_p}config.json") as f:
	cfg_data = json.load(f)


# Create the model
model_cfg = CustomModCon(hidden_size=cfg_data["hidden_size"])
gen_class = WrapClassModel(model_cfg)


# Load the weights
state_dict = load_file(f"{gen_class_p}model.safetensors", device="cpu")
gen_class.load_state_dict(state_dict)


gen_class.to(device)
gen_class.eval()


#endregion Classifier load



#region Data Loader


with open(mf_labs_p,"rb") as f:
	mf_labs= torch.load(f)

with open(task_labs_p,"rb") as f:
	task_labs = pickle.load(f)

#?Loading a TensorDataset version of our data
with open(tens_ds_p,"rb") as f:
	tens_ds = torch.load(f,weights_only=False)


loader = torch.utils.data.DataLoader(
	tens_ds,
	batch_size=batch_size,
	num_workers=4, #Start at this value, later we can move higher up - generally up to number of cores.
	prefetch_factor=8, #Decent to start here - how many batches are loaded at once
	persistent_workers=True,

)


diter = iter(loader)


#endregion Data Loader



#region INLP Transform Load

#region? (P,rowspace_projections,Ws)
	#? P is the final transform that is orthogonal to all the W-bias-directions
	#? rowspace_projections project onto a bias direction
	#? Ws are the weights for each of the rowspace projections
#endregion? (P,rowspace_projections,Ws)



with open(inlp_trained_p,"rb") as f:
	P,rowspace_p,Ws = pickle.load(f)
	inlp_trained = torch.Tensor(P).to(torch.float32)



with open(inlp_calced_p,"rb") as f:
	inlp_calced= torch.load(f,weights_only=False)
	inlp_calced = inlp_calced.to(torch.float32)	



#endregion INLP Transform Load



#region Main 


#region*# Real Inputs

# batch = next(diter)
# # batch = next(diter)
# token_type_ids = torch.zeros(batch[0].size())
# #Reconstructing the BatchEncoding object's dictionary manually.
# inputs = {"input_ids":batch[0].type(torch.int64).to(device),"token_type_ids":token_type_ids,"attention_mask":batch[1]}


#endregion*# Real Inputs



#region* Test inputs


bat = next(diter)

#? Tripple up the inputs with torch.cat
nbat = (torch.cat([bat[0],bat[0],bat[0]]),torch.cat([bat[1],bat[1],bat[1]]))

token_type_ids = torch.zeros(nbat[0].size())
#Reconstructing the BatchEncoding object's dictionary manually.
inputs = {"input_ids":nbat[0].type(torch.int64).to(device),"token_type_ids":token_type_ids,"attention_mask":nbat[1]}



#endregion* Test inputs





#region* Manipulating input
#? Since we're only testing INLP right now, this isn't too hard

with torch.inference_mode():
	with model.trace(inputs) as tracer:
		out6 = submodule_ref6.nns_output.save()

		to_inlp = submodule_ref6.nns_output[0]

		#region* Steering stuff
		
		inlp_control_set = to_inlp[0:10,:,:].to("cpu").clone().save()
		inlp_calced_set = to_inlp[10:20,:,:].to("cpu").clone().save()
		inlp_trained_set = to_inlp[20:,:,:].to("cpu").clone().save()

		#? Expecting the calculated one to work worse than the trained one.
		calc_res = inlp_calced_set @ inlp_calced
		train_res = inlp_trained_set @ inlp_trained


		#endregion* Steering stuff


		# recombs = torch.cat([inlp_control_set,calc_res,train_res])
		recombs = torch.cat([inlp_control_set,calc_res,train_res])
		submodule_ref7.input = recombs

		sub11 = submodule_ref_11.nns_output.save()
		finl = model.output.save()


#endregion* Manipulating input




#region Analysis

fin_out = finl[0]
out11 = sub11[0]


o11_a = out11[0:10,:,:]
o11_b = out11[10:20,:,:]
o11_c = out11[20:,:,:]



#region* "Decoding" output

#Section out the different experiments
set_a = fin_out[0:10,:,:]
set_b = fin_out[10:20,:,:]
set_c = fin_out[20:,:,:]




#Argmax to predict
for i in range(10):
	todec_a = torch.argmax(set_a,dim=-1)
	todec_b = torch.argmax(set_b,dim=-1)
	todec_c = torch.argmax(set_c,dim=-1)


#Decode with tokenizer
dec_a = []
dec_b = []
dec_c = []

for i in range(10):
	dec_a.append(tok.decode(todec_a[i]))
	dec_b.append(tok.decode(todec_b[i]))
	dec_c.append(tok.decode(todec_c[i]))


#endregion* "Decoding" output



#region curr test gen class


genclass_out_a = []
genclass_out_b = []
genclass_out_c = []

for i in range(len(set_a)):
	
	
	genclass_out_a.append(gen_class(torch.mean(o11_a[i],dim=0).to("cuda")))
	genclass_out_b.append(gen_class(torch.mean(o11_b[i],dim=0).to("cuda")))
	genclass_out_c.append(gen_class(torch.mean(o11_c[i],dim=0).to("cuda")))

#? Help print the outputs.
def gc_prntr(x):
	print("A")
	print(genclass_out_a[x])
	print("B")
	print(genclass_out_b[x])
	print("C")
	print(genclass_out_c[x])
	print("lab")
	print(mf_labs[x])
	print("#"*20)


print("STOP")
print("STOP")

#endregion curr test gen class







#endregion Analysis






#endregion Main























