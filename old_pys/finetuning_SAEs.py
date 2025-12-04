import pandas as pd
import pickle
import torch
from tqdm import tqdm

#Helpful for diagnostics
import pyperclip as clip
from collections import Counter


from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


#? Info
#region
#?Separate code for finetuning pretrained SAEs on the Asylex dataset
#?
#?
#endregion


#*Paths 
#region

lex_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/"

train_path = lex_p + "our_train.pkl"
test_path = lex_p + "our_test.pkl"

#endregion


#*Defs
#region

# device = "cuda"
device = "cpu"
model_name = "bert-base-cased"

#endregion


#*Dataset loading
#region

with open(train_path,"rb") as f:
    train_dat = pickle.load(f)

with open(test_path,"rb") as f:
    
    test_dat = pickle.load(f)

#endregion


#*Defining classes
#region

class AsylexFinetuneDataset(Dataset):
    
    def __init__(self, txt_list):
        # super().__init__() #?Abstract class, so nothing to init from super

        self.txt_list = txt_list


    def __len__(self):

        return len(self.txt_list)
        
    def __next__(self):
        pass




#endregion



#?Main code


#* Testing MaxAct stuff
#region

p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/maxact/"

fs= ["maxact_sparse_3_7000.pkl","maxact_sparse_6_7000.pkl","maxact_sparse_9_7000.pkl",]

maxacts= []

for f in fs:
    pf = p + f

    with open(pf,"rb") as f:
        maxacts.append(pickle.load(f))

print("STOP")
print("STOP")

#endregion  










