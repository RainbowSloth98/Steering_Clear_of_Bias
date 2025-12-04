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

#? Looking at the different versions of BTK SAEs to see which performed best.
	#* That would probably be the 144000 version, that has a a FVE of 0.9628

#endregion? Info


#region Classes and Objects

StampedLog = namedtuple("StampedLog",["step","l0","l2","fve","auxk_loss","loss"])


#endregion Classes and Objects



#region Path and loading

tmap = {1:"10k",2:"30k",3:"60k",4:"150k"}

ldeds = []

# 4 experiments
for i in range(1,5):
	p = f"/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/SAE_trainer/saves/tries/try_{i}/pretrain_SAEs/SAE6_logs/pretrain_log_SAE6.pkl"

	with open(p,"rb")as f:
		ldeds.append(pickle.load(f))


#endregion paths



#region Main Test area


keys = []
exs = []

for i in range(4):
	keys.append(sorted(list(ldeds[i].keys()))[-1])
	exs.append(ldeds[i][keys[i]])


def p_inter(k = len(exs)):
	for i in range(k):
		print("For Example: " + tmap[i+1])
		print("\tFve: "+str(exs[i].fve))
		print("\tLoss: " + str(exs[i].loss))
		print("#"*50)

def p_step(ex):
	ks = list(ex.keys())
	for i in range(len(ks)):
		print("For step: " + str(ks[i]))
		print("\tFve: "+str(ex[ks[i]].fve))
		print("\tLoss: " + str(ex[ks[i]].loss))
		print("#"*50)


print("STOP")
print("STOP")



#endregion Main Test area

