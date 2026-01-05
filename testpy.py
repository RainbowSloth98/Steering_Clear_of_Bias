import torch
import pickle
from tqdm import tqdm
import os
import torch
import pickle
from collections import namedtuple
from functools import partial

from datasets import load_dataset

from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.trainers import BatchTopKTrainer
from nnsight import LanguageModel



#region? Info

#* Looking at the different versions of BTK SAEs to see which performed best.
	#* That would probably be the 144000 version, that has a a FVE of 0.9628

#? Searching for gender features in 144k.

#endregion? Info


#region! Fix

#TODO Takes far too long to load all the features. (even a subset)

#endregion! Fix


#region Classes and Objects



#endregion Classes and Objects



#region Path and loading



#region*# T1: Looking through different BTK SAEs


# tmap = {1:"10k",2:"30k",3:"60k",4:"150k"}

# ldeds = []

# # 4 experiments
# for i in range(1,5):
# 	p = f"/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/SAE_trainer/saves/tries/try_{i}/pretrain_SAEs/SAE6_logs/pretrain_log_SAE6.pkl"

# 	with open(p,"rb")as f:
# 		ldeds.append(pickle.load(f))


#endregion*# T1: Looking through different BTK SAEs



#region* T2: Load feats to look for gender

batch1 = list(range(3072))

feats = []

for ind in tqdm(batch1,desc="Loading feats..."):
	p = f"/home/strah/Documents/Work_related/thon-of-py/blober/features/feature_{ind}.pkl"

	if not (os.path.exists(p)):
		pass
	else:
		with open(p,"rb") as f:
			feats.append(pickle.load(f))



#endregion* T2: Load feats to look for gender




#endregion Path and loading




#region Main Test area







print("STOP")
print("STOP")


#region*# Mini-notes

#f["examples][2][example_idx]

#endregion*# Mini-notes



#endregion Main Test area

