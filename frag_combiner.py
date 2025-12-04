import pandas as pd
import pickle
from tqdm import tqdm
import gc
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from nnsight import LanguageModel
from torch.utils.data import TensorDataset



#region Params


#region* inits

model_name = "BAAI/bge-large-en-v1.5"
device = "cuda"
batch_size = 1
seq_len = 512
total_examples = 23121 #? Should always be == len(lab_chunks_ds)
batch_size=63 #? Should be a divisor of total_examples
run_for = total_examples//batch_size
save_steps = [i for i in range(1,run_for+1)]

#endregion


#region? adjustable



# tok_type = "cls"
tok_type = "pool"
tok_map = {"cls":"CLS", "pool":"POOL"}

model_type = "bge"
# model_type = "bert"

#endregion


#endregion



#region Paths

if (model_type == "bge"):
    ogroot_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/bge-large-en/"
elif (model_type == "bert"):
	ogroot_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/"


root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/frag_combo/"

save_p = root_p + "saves/"
cls_sp = save_p + "CLS/"
pool_sp = save_p + "POOL/"



load_p = ogroot_p + f"saves/frag_saves/{tok_map[tok_type]}/"

ld_namer = "cls_save_"

#endregion



#region Main Creating

#? Run the commented out below to get what we need

#endregion





#region# Combining activations.


save_steps.pop(-1) #? Make sure this is not in the loop

# #? f_c stands in for frag_choice.
for f_c in range(2,6):

	cls_acts = []
	for s in save_steps:

		lp = load_p + f"frag_v{f_c}/" + f"cls_save_{s}.pt"

		with open(lp,"rb") as f:
			cls_acts.append(pickle.load(f))



	all_cls = torch.cat(cls_acts)



	if(tok_type == "cls"):
		sp = cls_sp + f"all_cls_acts_v{f_c}.pt"

	elif(tok_type == "pool"):
		sp = pool_sp + f"all_cls_acts_v{f_c}.pt"


	with open(sp,"wb") as f:
		torch.save(all_cls,f)


# print("STOP")
# print("STOP")

#endregion


#region# Create and Save new labels

#? The old tensors don't work, so we need to create new ones.

#region? Method for finding the missing indicies

#? Take the original indicies, and do follwoing
#? n and k start at 0. Find an example that is not in line with indicies.
#? Note the missing candidate, change n to the index that was out of line
#? and set k to be the difference between the value at the index, and the index itself.

# for i in range(n,len(og_inds)):
#     if (og_inds[i]-k != i):
#         print(i)
#         break

#endregion


#? Manually found the 23 labels that were extra.
# labels_to_rem = [
# 	3100,3103,3137,
# 	3222,4797,4823,
# 	4845,4958,5032,
# 	5034,5035,5096,
# 	5164,5185,5200,
# 	5528,5742,6008,
# 	11372,12850,18401,
# 	19012,19368,
# ]


# new_labels = [train_labels[i] for i in range(len(train_labels)) if i not in labels_to_rem]

# nlabs = torch.stack(new_labels)

#region*# Saving

# nlab_sp = root_p + "train_labels_23121.pt"
# with open(nlab_sp,"wb") as f:
# 	torch.save(nlabs,f)

#endregion

#endregion


#region# Labels and Data to full dataset

# rex = list(og_data["raw_txt"])


# tens_doc_list = [chunk_document(rex[i],tok,) for i in range(len(rex))]

# with open(load_p+"tens_doc_list.pkl","rb") as f:
# 	tens_doc_list= pickle.load(f)


# #Create data input for dataset
# chunk_dat = torch.cat(tens_doc_list)

#region*# Save above because it take long

# # ps= "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/temp_saves/tens_doc_list.pkl"
# # with open(ps,"wb") as f:
# # 	pickle.dump(tens_doc_list,f)

#endregion


# chunk_lens = [tens_doc_list[i].size(0) for i in range(len(tens_doc_list))]
# chunk_lens = torch.Tensor(chunk_lens).to(torch.int64)


# chunk_labels = torch.repeat_interleave(new_labels,chunk_lens)

# # Create label input for dataset
# chunk_labels = chunk_labels.unsqueeze(-1)

# nnsight_chunk_ds_obj = TensorDataset(chunk_dat,chunk_labels)


#endregion


print("END FILE")