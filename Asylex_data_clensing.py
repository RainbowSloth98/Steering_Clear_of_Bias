#Standard imports
import ast
from datasets import load_dataset
import pandas as pd
import pickle
import os
from tqdm import tqdm
import re
import more_itertools
import gc

#For streaming large data
import json
from pathlib import Path


import torch
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments


from transformer_lens import HookedEncoder, HookedTransformer
from transformers import PreTrainedModel,PretrainedConfig


#Helpful for diagnostics
import pyperclip as clip
from collections import Counter



#region? Info on Asylex
#?############################################
#? Documents
#* 	59_112 total
#* 	From Canada bewtween 1996-2022
#? Tasks:
#* 	Entity extraction for legal search
#* 	Judgement precition (of refugees)
#? Task 1
#* 	19_115 human labelled annotations
#* 	76_523 golden (gt?) annotations total, rest were rule based
#* 		20 legally relevant entry types to deal with
#* 		Or is it 22, they mention this later on in the paper?
#* 	6_278_028 "silver-standard" labelled samples inferred via n-ER model
#* 		Silver-standard are LLM generated labels (for case outcomes)
#* 		Gold-standard are manually labelled annotations
#? Task 2
#* 	1682 labelled docs for outcomes
#* 	30_944 training docs
#?############################################
#endregion? Info on Asylex






#region Paths

cache_path = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/cache_ds/"
og_datroot_path = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/AsyLex_dataset/"
outcome_path = og_datroot_path + "outcome_train_test/"


id_data_path = og_datroot_path + "determination_label_extracted_sentences.csv"
BERT_test_path = outcome_path + "test_dataset_gold.csv"
BERT_train_path = outcome_path + "train_dataset_silver.csv"

#endregion Paths




#region Funcs


#region* UUtils

def uutil_fix_years_xxx_x(input_string):
    # Regex to match years with spaces in the middle (e.g., "201 6")
    pattern = r'(\d{3})\s(\d{1})'  # Matches "201 6" or similar patterns
    # Replace the matched pattern with the correct year format
    fixed_string = re.sub(pattern, r'\1\2', input_string)
    
    # You can add more patterns here if needed (for other specific cases).
    return fixed_string


def uutil_fix_years_in_string(input_string):
    # Regex to match any 4-digit year with spaces in between
    pattern = r'(\d)\s*(\d)\s*(\d)\s*(\d)'  # Matches any space within a 4-digit year
    # Replace the matched pattern with the correct year format
    fixed_string = re.sub(pattern, r'\1\2\3\4', input_string)
    
    return fixed_string


def uutil_flatten(mixed_list):
    for item in mixed_list:
        if isinstance(item, list):
            yield from uutil_flatten(item)  # Recursively flatten sublists
        else:
            yield item  # Yield non-list items

#endregion* UUtils


#region* Utils

#? Wrapper for the other generator func
def util_flatten_list(l):
    return list(uutil_flatten(l))


def util_find_all_years(text):
    # Find all 4-digit years between 1996 and 2022
    matches = re.findall(r'\b(199[6-9]|20[0-1][0-9]|202[0-2])\b', text)

    ret = [int(year) for year in matches]

    if(len(ret) == 1):
        return ret[0]
    else:
        print("Was not only one!")
        return ret


#endregion* Utils


# Used to fix the wrong dates in cc_ents 
def fix_ents(ser, probs):

    all_dates = ser["extracted_dates"]

    #Since the dates are all together into one string, we need to separate them by ',
    dates = all_dates.split("',")


    #Now, we extract all the years that are between 1996 and 2022 for all of them

    extracted = []

    for date in dates:

        #Accounts for misformatting, like 201 6, or 20 06. Can use the more restricted version too.
        date = uutil_fix_years_in_string(date)

        extracted.append(util_find_all_years(date))
        pass

    if not (all(isinstance(item,int) for item in extracted)):
        
        print("entered the if for flattening")
        #Flatten into one list
        extracted = util_flatten_list(extracted)

        #Making sure that all of the dates are in int format for the next step
        extracted = list(map(lambda x: int(x), extracted))

    

    extracted = list(set(extracted))

    cands = [[y,id,str(y)+"canlii"+str(id)+".txt"] for y in extracted]

    checked = []

    for i in range(len(cands)):

        pth = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/AsyLex_dataset/cases_anonymised_raw_text/" + cands[i][2]
        
        if(os.path.exists(pth)):
            checked.append(cands[i])


    if(len(checked) == 0):
        probs.append(id)

    else:
        #Set it to the first of the ones that was checked to work
        ser["DATE_DECISION"] = checked[0][0]


    return ser


#endregion Funcs



#region Loads

id_data = pd.read_csv(id_data_path, delimiter=";")
og_train = pd.read_csv(BERT_train_path, delimiter=";")
og_test = pd.read_csv(BERT_test_path, delimiter=";")

with open(cache_path+"full_data.pkl","rb") as f:
    my_clean_data = pickle.load(f)

#Adding the raw data and case cover stuff to the test and train
my_test = pd.merge(og_test,my_clean_data,on=["decisionID", "decision_outcome"],how="left")
my_train = pd.merge(og_train,my_clean_data,on=["decisionID", "decision_outcome"],how="left")

#endregion Loads


#region Main Code



#*determination sentence ids are not unique for some reason. Lets make them unique
unqs = id_data.drop_duplicates()

#*Has a single bad entry.
unqs = unqs.drop(30585)

#*IDs are stored as str instead of int...
unqs["decisionID"] = unqs["decisionID"].astype(int)


#region*# From the step where we were looking for uniques

# unq_s = set(unqs["decisionID"])
# id_to_df = {uid: unqs[unqs["decisionID"] == uid] for uid in unq_s}

#endregion*# From the step where we were looking for uniques


#*Removes duplicates by ID
clean_determs = unqs.drop_duplicates(subset='decisionID', keep='first')


#*Marging the clean determination sentences with the rest of the data
fin_train = pd.merge(my_train,clean_determs,on = "decisionID", how="left")
fin_test = pd.merge(my_test,clean_determs,on = "decisionID", how="left")


#*Apparently ~400 of the test entries don't correlate to any data entry...
ndate1 = fin_test[fin_test["extracted_sentences_determination"].isna() == True]
tst_c = fin_test[~fin_test.isin(ndate1.to_dict(orient="list")).all(axis=1)]


#*Do the same for the train set, thankfully way fewer of these are problematic
ndate2 = fin_train[fin_train["extracted_sentences_determination"].isna() == True]
trn_c = fin_train[~fin_train.isin(ndate2.to_dict(orient="list")).all(axis=1)]

#*Needed to do this because, for some reason, 758 entries in the test set are ALSO in the train set...
inter = pd.merge(trn_c,tst_c,how="inner")
trn_c2 = trn_c[~trn_c.isin(inter.to_dict(orient="list")).all(axis=1)]

#*Need to remove the extra class that exists only in the train set
rems = trn_c2[trn_c2["decision_outcome"] == 2]
trn_c3 = trn_c2.drop(rems.index)

#*We should probably rename the columns...
rename = {'decisionID' : 'id','decision_outcome' : 'outcome','DATE_DECISION' : 'date','real_id' : 'file_id','extracted_sentences_determination' : 'basis'}
trn = trn_c3.rename(columns=rename)
tst = tst_c.rename(columns=rename)

#*Also make sure that ids are ints and not floats
tst["id"] = tst["id"].astype(int)

#*Reset the index such that we can use .loc
trn = trn.reset_index()
tst = tst.reset_index()

#*Also need adjust the data, since its stored ["some string", "mb another one"] instead of a single string...
trn_dat = list(trn["basis"])
tst_dat = list(tst["basis"])



#region? Transforming "basis" from string list into just list.

# n_trn = [] #new train
# n_tst = [] #new test


# for i in range(len(trn_dat)):
#     s = ""
#     t = ast.literal_eval(trn_dat[i]) #? Changing "["determ1","determ2"]" into just a list
#     for j in range(len(t)): #? Joining it up into a single string
#         s += " " + t[j]
#     n_trn.append(s)

#* Doing the same as above
# for i in range(len(tst_dat)):
#     s = ""
#     t = ast.literal_eval(tst_dat[i])
#     for j in range(len(t)):
#         s += " " + t[j]
#     n_tst.append(s)

# trn["basis"] = n_trn
# tst["basis"] = n_tst


#region**# Saving

# with open(train_path,"wb") as f:
#     pickle.dump(trn,f)

# with open(test_path,"wb") as f:
#     pickle.dump(tst,f)

#endregion**# Saving

#endregion? Unlabelled region 1




#region? Finalisation
#* Afterwards we converted tensors to lists of strings, and fed them to bert
#* This left us with saved activations for all of the layers, of which we saved
#* the ones that we needed, namely CLS tokens for class, and post_resid 3,6,9 for SAEs
#* We also stored the labels as well.
#endregion? Finalisation




#endregion Main Code








print("End file")