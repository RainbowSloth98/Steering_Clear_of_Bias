import pandas as pd
import torch
import pickle




#? Info
#region
#?File that takes in the already tokenized dataset, and returns a new one that is segmented into 512 sections
#?
#?

#endregion




#* Params
#region



#endregion 



#* Paths
#region

rootds_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/"

ds_p = rootds_p + "our_train.pkl"
tok_ds_p = rootds_p + "toked_lex.pkl"


#endregion  


#* Defs
#region



#endregion



#* Funcs
#region



#endregion


#* Main
#region


with open(tok_ds_p,"rb") as f:
    tok_ds = pickle.load(f)




print("STOP")
print("STOP")

sqlen = 512
sep_tok_ds = [[doc[p:p+sqlen] for p in range(0,len(doc),sqlen)] for doc in tok_ds]





#endregion  




