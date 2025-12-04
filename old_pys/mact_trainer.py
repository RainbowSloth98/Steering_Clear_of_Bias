from tqdm import tqdm
import os
import torch
import pickle
from collections import defaultdict, Counter
import gc

from datasets import load_dataset

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from nnsight import LanguageModel

import random



#region? Info

#? Main program for finding feature activation examples.
#? This version implements a "Hierarchical Sampling and Counting" algorithm to
#? efficiently gather word-frequency distributions and example contexts for each feature/stratum.

#! New SAE trained 25.11.25
#? This one is BTK, and we want to find features that are in a range as well.

#endregion? Info



#region Params


#region* Non-test


#region? trn_stage
trn_stage = "pretrain"
# trn_stage = "finetune"
#endregion? trn_stage


#region? sae_type
# sae_type = "Standard"
sae_type = "BTK"
#endregion? sae_type


#endregion* Non-test







#endregion Params



#region Funcs and classes

# This function's only job is to generate a context string for a given token index.
def get_token_context_string(all_toks, tokenizer, token_idx, context_size):
	"""Generates a string of text context around a specific token."""
	# Ensure all_toks is a 1D tensor or list for this function
	if all_toks.ndim > 1:
		all_toks = all_toks.squeeze()
	ctx_s = max(0, token_idx - context_size)
	ctx_e = min(len(all_toks), token_idx + context_size + 1)
	return tokenizer.decode(all_toks[ctx_s:ctx_e])



#Used to create a data loader from the Asylex dataset
class AsyLexDataset(torch.utils.data.Dataset):

	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return {"text":self.data[i]}



#endregion Funcs and classes	



#region Defs

#------------------BERT-------------------
seq_len = 512
model_name = "bert-base-uncased" #! UNCASED
device = "cuda"
act_dim = 768

#? Use to be 2^8 for mact, now less due to strata
# batch_size = 2**5 
# batch_size = 1 #TODO find batch_size
batch_size = 2 #Start testing at 2
#* ------------------SAE--------------------
#? This controls surprisingly many things
# lyndex = [3, 6, 9] #TODO Generalise lyndex
lyndex = [6] 
check_point = 144000
exp_factor = 8
dict_size = act_dim*exp_factor
btk_k = 350
num_saes = len(lyndex)
#* -----------------Mact--------------------
act_thresh = 0.2 #? This was selected rather arbitrarily
tok_ctx = 20 #Token context
#* ------------------Strata-----------------
RES_SIZE = 3 #Size of reservoir
strata_bands = [(0.1, 0.4999), (0.5, 0.9999), (1, 1.9999), (2, 2.9999), (3, 4.4999) , (4.5, 6.9999),]
num_strata = len(strata_bands)
#* -------------------Saving-----------------
#TODO Find steps and spnum.
steps = 23144 #might be 22299?
#Save point number; number of saves 
spnum = 20 
total_batches = steps // batch_size
save_batch_period = total_batches // spnum if spnum > 0 else total_batches + 1

#TODO Investigate this more
save_batches = {int(i * save_batch_period) for i in range(1, spnum + 1)}

#TODO Find number of saves
CHECK_EVERY_N_BATCHES = 2 
# CHECK_EVERY_N_BATCHES = 30



#endregion Defs



#region Paths

root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/mact/"
loads_p = root_p + "loads/"
saves_p = root_p + "saves/"

lex_p = loads_p + "our_train.pkl"


#region* Save Path Defs
sae_cp = [check_point, check_point, check_point]
sae_ps = []
save_data_ps = []
for i in range(len(lyndex)):
	sae_ps.append(loads_p + f"{sae_type}_SAE{lyndex[i]}_{sae_cp[i]}.pt") #BTK_sae_144000.pt
	save_data_ps.append(saves_p + f"maxact_sparse_{lyndex[i]}_{sae_cp[i]}.pkl") #maxact_sparse_9_7000.pkl
#endregion* Save Path Defs


#endregion Paths



#region TODO SAE and Sampling Defs

saes = [BatchTopKSAE.from_pretrained(p, device=device) for p in sae_ps]
for sae in saes:
	sae.eval()


#curr IMPORTANT; Cannot save because of this. Check below.
# Function to initialize our data structures
def initialize_storage():
    print("Initializing/Resetting data storage...")
    examples = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'examples': []}))
    counters = defaultdict(int)
    return examples, counters

feature_strata_examples, feature_strata_counters = initialize_storage()



#! This should apparently fix it.
# def make_word_entry():
#     return {'count': 0, 'examples': []}

# def make_token_dict():
#     return defaultdict(make_word_entry)

# def initialize_storage():
#     print("Initializing/Resetting data storage...")
#     examples = defaultdict(make_token_dict)
#     counters = defaultdict(int)
#     return examples, counters


#endregion SAE and Sampling Defs



#region Model

model = LanguageModel(model_name, device_map=device)
model.eval()

#Defining submodules
submodules = []
for i in range(len(lyndex)):
	submodules.append(eval(f"model.bert.encoder.layer[{lyndex[i]}]")) #TODO Check to see if this works

#endregion BERT



#region Tokenizer
tokenizer = model.tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.backend_tokenizer.enable_truncation(max_length=seq_len)
tokenizer.backend_tokenizer.enable_padding(length=seq_len, pad_id=tokenizer.pad_token_id, pad_token=tokenizer.pad_token)
#endregion



#region Dataset and Dataloader


with open(lex_p,"rb") as f:
	lex = pickle.load(f)
	lex_to_ds = list(lex["raw_txt"])

dataset = AsyLexDataset(lex_to_ds)

loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		num_workers=4,
		prefetch_factor=8,
		persistent_workers=True,
		drop_last=True,
	)


diter = iter(loader)

#endregion Dataset and Dataloader



#region Main Code


with tqdm(total=total_batches, desc="Getting Max Acts...") as loop:
	for batch_idx in range(total_batches):



	#region* Getting inputs

		batch = next(diter)
		inputs = tokenizer(batch["text"], return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)
		all_toks = inputs["input_ids"].to('cpu')
		inputs_gpu = {k: v.to(device) for k, v in inputs.items()}	
		
	#endregion* Getting inputs



		#region* Extracting Features
		
		with torch.inference_mode():
			with model.trace(inputs_gpu, invoker_args={"max_length": seq_len}) as tracer:

				#region**# Generalise to all

				# l3 = submodules[0].nns_output.save()
				# l6 = submodules[1].nns_output.save()
				# l9 = submodules[2].nns_output.save()

				# submodules[2].nns_output.stop()
				
				#endregion**# Generalise to all
				
				# l6 = submodules[1].nns_output.save() #TODO Uncomment when doing all
				l6 = submodules[0].nns_output.save() 

			# Shape: (num_saes, batch_size, seq_len, act_dim)
			# acts = torch.stack([l3[0], l6[0], l9[0]])
			acts = torch.stack([l6[0]]) #TODO Change when doing all SAEs

		#endregion* Extracting Features



		#region* Selection Logic


		#region** Get the activations for the current SAE

		#Define the cache outside
		context_cache = {}

		# for sae_idx in tqdm(range(num_saes),desc="SAEs..."):
		for sae_idx in range(num_saes): #TODO Remove when doing more than 1
			
			
			
			current_acts = acts[sae_idx]
			
			with torch.inference_mode():
				# Encode to get features for THIS SAE ONLY.
				# This is the large tensor that we are now creating and destroying in each loop iteration.
				feats_one_sae = saes[sae_idx].encode(current_acts)
				# Shape: (batch_size, seq_len, num_feats)

			current_batch_size = feats_one_sae.shape[0]


		#endregion** Get the activations for the current SAE



		#region** Stratafied Reservoiring

			# Efficiently find all activations within our defined bands for this batch
			for stratum_idx, (lower_bound, upper_bound) in enumerate(strata_bands):
				
				# Create a boolean mask for the current stratum
				mask = (feats_one_sae >= lower_bound) & (feats_one_sae < upper_bound)
				
				# Get indices of all activations in this batch that fall into the stratum
				# Shape of found_indices: (num_found, 3) where columns are (batch_item_idx, token_idx_in_seq, feat_idx)
				found_indices = torch.nonzero(mask)


				#region**# Previous caching stuff
				
				# #Find list of unique batch-token_position pairs
				# uniques = torch.unique(found_indices[:,:2],dim=0)

				# #Create a lookup table of contexts for each unique one, accessed by a key
				# context_cache = {tuple(u.tolist()):get_token_context_string(all_toks[u[0]], tokenizer, u[1], tok_ctx) for u in uniques}
				

				#endregion**# Previous caching stuff
				

				# Iterate through only the activations that we care about
				for idx_tuple in found_indices:

					batch_idx, seq_idx, feat_idx = idx_tuple.tolist()

					storage_key = (sae_idx,feat_idx,stratum_idx)

					#For each feature/strata combo, increment a counter
					feature_strata_counters[storage_key] += 1

					#Get the activating tok
					token_id = all_toks[batch_idx, seq_idx].item()

					#Find or create entry in defaultdict for this word
					word_data = feature_strata_examples[storage_key][token_id]

					#Count times this feat activated on this word. Building distribution across words.
					word_data["count"] += 1

					#region* Reservoir logic - defines should_add for check afterwards
					
					if len(word_data['examples']) < RES_SIZE:
						# Reservoir is not full, so we must generate and add the example.
						should_add = True
						slot_to_replace = -1 # Append
					else:
						# Reservoir is full, randomly decide whether to replace an existing example.
						j = random.randint(0, word_data['count'] - 1)
						should_add = j < RES_SIZE
						slot_to_replace = j

					#endregion* Reservoir logic for the examples

					#We decided to add; Expensive stuff happens here
					if should_add:
						
						#Create a key to access contexts for the same words. No feat_idx as that's already sorted above
						context_key = (batch_idx,seq_idx)

						#Part of memoization. Don't reuse the same stuff
						if context_key not in context_cache:
							context_cache[context_key] = get_token_context_string(all_toks[batch_idx], tokenizer, seq_idx, tok_ctx)


						#Get all of the activations
						full_feature_vector = feats_one_sae[batch_idx, seq_idx, :]

						#Create a mask to keep only the active ones; We can recreate later
						active_mask = full_feature_vector > act_thresh

						current_metadata = {
							'value': feats_one_sae[batch_idx, seq_idx, feat_idx].item(),
							'context': context_cache[context_key],
							'token_id': token_id,
							'step': batch_idx * current_batch_size + batch_idx,
							'sparse_features': (torch.nonzero(active_mask).squeeze(-1).cpu(), full_feature_vector[active_mask].cpu())
						}

						#* Either replace old, or add new.
						if slot_to_replace != -1: # Replace existing example
							word_data['examples'][slot_to_replace] = current_metadata
						else: # Append new example
							word_data['examples'].append(current_metadata)


		#endregion** Stratafied Reservoiring



		#endregion* Selection Logic



		#region* Saving
		
		# We check at the end of the batch, using `batch_idx + 1`
		if (batch_idx + 1) % CHECK_EVERY_N_BATCHES == 0:
			print(f"\n--- Checkpoint at Batch {batch_idx+1} ---")
			
			# Save the current chunk of data
			print("Saving current data chunk...")
			for i in range(num_saes):
				sae_examples = {k: v for k, v in feature_strata_examples.items() if k[0] == i}
				sae_counters = {k: v for k, v in feature_strata_counters.items() if k[0] == i}
				
				data_to_save = {
					'examples_and_word_counts': sae_examples, 
					'total_strata_counts': sae_counters,
				}
				

				# Create a unique filename for this chunk
				chunk_save_path = f"{save_data_ps[i]}_batch_{batch_idx+1}.pkl"
				with open(chunk_save_path, "wb") as f:
					pickle.dump(data_to_save, f)
			print(f"Chunk saved to {chunk_save_path}")

			# Reset the data structures to clear memory
			feature_strata_examples, feature_strata_counters = initialize_storage()
			gc.collect() # Ask the garbage collector to free up memory

		#endregion* Saving



		#region* Release memory
		
		del acts
		del feats_one_sae 
		del current_acts # This is only a slice/view of acts, but good practice
		del inputs_gpu # The model inputs on the GPU
		del context_cache # The cache is only needed for the current batch
		gc.collect() 
		torch.cuda.empty_cache()
		
		#endregion* Release memory

		loop.update(1)

#endregion Main Code











#region Unused


#region*# trn_stage data loading

# if (trn_stage == "pretrain"):
# 		# Use a smaller, non-streaming dataset for demonstration
# 		dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# 		loader = torch.utils.data.DataLoader(
# 			dataset,
# 			batch_size=batch_size,
# 			num_workers=4,
# 			prefetch_factor=8,
# 			persistent_workers=True,
# 		)

# elif (trn_stage == "finetune"):

# 	with open(lex_p,"rb") as f:
# 		lex = pickle.load(f)
# 		lex_to_ds = list(lex["raw_txt"])


# 	dataset = AsyLexDataset(lex_to_ds)


# 	loader = torch.utils.data.DataLoader(
# 			dataset,
# 			batch_size=batch_size,
# 			num_workers=4,
# 			prefetch_factor=8,
# 			persistent_workers=True,
# 			drop_last=True,
# 		)

#endregion*# trn_stage data loading





#endregion Unused


print("END FILE")
