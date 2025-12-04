from tqdm import tqdm
import os
import torch
import pickle
from collections import defaultdict, Counter
import gc
import random

from datasets import load_dataset
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from nnsight import LanguageModel



#region? Info
#? Main program for finding feature activation examples.
#? V3 Update: Optimized for "Feature-Centric" storage.
#? - Stores 3 distinct strata (Low, Mid, High).
#? - Caps examples at ~7 per stratum (Fixed size storage).
#? - Maintains global word freq counts per feature.
#endregion? Info





#region Params

#region* Non-test
trn_stage = "pretrain"
sae_type = "BTK"
#endregion* Non-test

#endregion Params



#region Funcs and classes

# Replace your current get_token_context_string with this:

def get_token_context_string(all_toks, tokenizer, token_idx, context_size):
    """
    Generates a string of text context around a specific token,
    highlighting the specific token that activated.
    """
    if all_toks.ndim > 1:
        all_toks = all_toks.squeeze()
        
    # 1. Define bounds
    ctx_s = max(0, token_idx - context_size)
    ctx_e = min(len(all_toks), token_idx + context_size + 1)
    
    # 2. Decode in three parts: Prefix, Target, Suffix
    # We use 'clean_up_tokenization_spaces=False' to have tighter control, 
    # though standard decoding is usually fine.
    
    prefix = tokenizer.decode(all_toks[ctx_s:token_idx])
    
    # We highlight the target. 
    # Note: If it's a subword like '##ing', this will show '|##ing|'
    # This is GOOD for interpretability (precision).
    target_token = tokenizer.decode(all_toks[token_idx:token_idx+1])
    
    suffix = tokenizer.decode(all_toks[token_idx+1:ctx_e])
    
    # 3. Stitch together with highlighting
    # You can change '|' to '>>' or '*' or '<b>' as preferred.
    return f"{prefix} |{target_token}| {suffix}"


# def get_token_context_string(all_toks, tokenizer, token_idx, context_size):
# 	"""Generates a string of text context around a specific token."""
# 	if all_toks.ndim > 1:
# 		all_toks = all_toks.squeeze()
# 	ctx_s = max(0, token_idx - context_size)
# 	ctx_e = min(len(all_toks), token_idx + context_size + 1)
# 	return tokenizer.decode(all_toks[ctx_s:ctx_e])


class AsyLexDataset(torch.utils.data.Dataset):
	def __init__(self, data):
		self.data = data
	def __len__(self):
		return len(self.data)
	def __getitem__(self, i):
		return {"text":self.data[i]}

#! CHANGE: New Data Structure Definitions
# We want a flat structure per feature.
def create_feature_entry():
	return {
		# Tracks which words trigger this feature (across all intensities)
		'word_counts': Counter(), 
		
		# The storage buckets for our examples
		# 0: Low, 1: Mid, 2: High
		'examples': {0: [], 1: [], 2: []}, 
		
		# Internal counters to make reservoir sampling work (n seen so far)
		'strata_seen': {0: 0, 1: 0, 2: 0} 
	}



def create_sae_dict():
	return defaultdict(create_feature_entry)



#endregion Funcs and classes	



#region Defs

#------------------BERT-------------------
seq_len = 512
model_name = "bert-base-uncased" 
device = "cuda"
act_dim = 768

#! CHANGE: Batch Size
# Since we are no longer storing massive dictionaries, memory is freed up.
# You can likely push this to 16 or 32 depending on GPU VRAM.

batch_size = 16

#* ------------------SAE--------------------
lyndex = [6] 
check_point = 144000
exp_factor = 8
dict_size = act_dim*exp_factor
btk_k = 350
num_saes = len(lyndex)

#* -----------------Mact--------------------
act_thresh = 0.2 
tok_ctx = 20 
MAX_STORED_FEATURES = btk_k 

#* ------------------Strata-----------------
#! CHANGE: Simplified Strata
# We now have 3 buckets. 
# You should tune the boundaries based on your SAE.
# e.g., Low (0.2-1.5), Mid (1.5-4.0), High (4.0+)
strata_bands = [
	(0.2, 1.4999),   # Index 0: Low
	(1.5, 3.9999),   # Index 1: Mid
	(4.0, 9999.0)    # Index 2: High
]
EXAMPLES_PER_STRATUM = 7 # We want ~7 examples per band (Total 21 per feature)

#* -------------------Saving-----------------
steps = 23144 
spnum = 20 
total_batches = steps // batch_size
save_batch_period = total_batches // spnum if spnum > 0 else total_batches + 1
save_batches = {int(i * save_batch_period) for i in range(1, spnum + 1)}

# We can save less frequently now that RAM isn't exploding
CHECK_EVERY_N_BATCHES = 2

#endregion Defs



#region Paths


root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/mact/"
loads_p = root_p + "loads/"
saves_p = "/media/strah/344A08F64A08B720/Work_related/store/new_max_acts/"
# saves_p = root_p + "saves/"
lex_p = loads_p + "our_train.pkl"

sae_cp = [check_point, check_point, check_point]
sae_ps = []
save_data_ps = []
for i in range(len(lyndex)):
	sae_ps.append(loads_p + f"{sae_type}_SAE{lyndex[i]}_{sae_cp[i]}.pt") 
	save_data_ps.append(saves_p + f"maxact_v3_{lyndex[i]}_{sae_cp[i]}")


#endregion Paths



#region SAE Init

saes = [BatchTopKSAE.from_pretrained(p, device=device) for p in sae_ps]
for sae in saes:
	sae.eval()

def initialize_storage():
	print("Initializing/Resetting data storage...")
	# Dictionary structure: [SAE Index][Feature Index] -> FeatureEntry
	storage = defaultdict(create_sae_dict)
	return storage

# Main storage object
feature_storage = initialize_storage()


#endregion



#region Model

model = LanguageModel(model_name, device_map=device)
model.eval()
submodules = []
for i in range(len(lyndex)):
	submodules.append(eval(f"model.bert.encoder.layer[{lyndex[i]}]"))

#endregion



#region Tokenizer
tokenizer = model.tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.backend_tokenizer.enable_truncation(max_length=seq_len)
tokenizer.backend_tokenizer.enable_padding(length=seq_len, pad_id=tokenizer.pad_token_id, pad_token=tokenizer.pad_token)
#endregion



#region Dataset

with open(lex_p,"rb") as f:
	lex = pickle.load(f)
	lex_to_ds = list(lex["raw_txt"])

dataset = AsyLexDataset(lex_to_ds)
loader = torch.utils.data.DataLoader(
		dataset, batch_size=batch_size, num_workers=4, prefetch_factor=8, persistent_workers=True, drop_last=True,
	)
diter = iter(loader)

#endregion






#region Main Code

with tqdm(total=total_batches, desc="Getting Max Acts (V3)...", colour="purple") as loop:
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
				l6 = submodules[0].nns_output.save() 
			acts = torch.stack([l6[0]]) 
		#endregion* Extracting Features


		#region* Selection Logic

		#region** Encoding acts to SAE feats
		
		for sae_idx in range(num_saes):
			current_acts = acts[sae_idx]
			with torch.inference_mode():
				feats_one_sae = saes[sae_idx].encode(current_acts)
			current_batch_size = feats_one_sae.shape[0]

		#endregion** Encoding acts to SAE feats



			#region** Pre-caching Contexts (Preserved from V2)

			context_cache = {}
			all_found_indices = []
			for lower, upper in strata_bands:
				mask = (feats_one_sae >= lower) & (feats_one_sae < upper)
				found_indices_for_stratum = torch.nonzero(mask)
				all_found_indices.append(found_indices_for_stratum)
			
			if all_found_indices:
				combined_indices = torch.cat(all_found_indices, dim=0)
				if combined_indices.numel() > 0:
					unique_token_positions = torch.unique(combined_indices[:, :2], dim=0)
					for u in unique_token_positions:
						b_idx, t_idx = u.tolist()
						context_cache[(b_idx, t_idx)] = get_token_context_string(all_toks[b_idx], tokenizer, t_idx, tok_ctx)

			#endregion** Pre-caching Contexts (Preserved from V2)

			# Iterate through our 3 Strata (Low, Mid, High)
			for stratum_idx, (lower_bound, upper_bound) in tqdm(enumerate(strata_bands),desc="Strata", colour="green",leave=False):
				
				mask = (feats_one_sae >= lower_bound) & (feats_one_sae < upper_bound)
				found_indices = torch.nonzero(mask)
				
				for idx_tuple in tqdm(found_indices,desc="Entries...",colour="blue",leave=False):
					b_idx, t_idx, feat_idx = idx_tuple.tolist()
					
					# 1. Retrieve the Feature Entry
					# This gives us access to this feature's global word counts and stratified examples
					feat_entry = feature_storage[sae_idx][feat_idx]
					
					token_id = all_toks[b_idx, t_idx].item()
					
					# 2. Always update the Global Word Count (Distribution)
					feat_entry['word_counts'][token_id] += 1
					
					# 3. Reservoir Sampling for Examples
					# We maintain a fixed size pool (e.g. 7) for this specific stratum.
					
					feat_entry['strata_seen'][stratum_idx] += 1
					current_seen = feat_entry['strata_seen'][stratum_idx]
					current_examples = feat_entry['examples'][stratum_idx]
					
					should_add = False
					slot_to_replace = -1
					

					#region* Reservoir logic
					
					if len(current_examples) < EXAMPLES_PER_STRATUM:
						should_add = True
					else:
						# Classic Reservoir logic: replace with probability k/n
						j = random.randint(0, current_seen - 1)
						if j < EXAMPLES_PER_STRATUM:
							should_add = True
							slot_to_replace = j
					
					#endregion* Reservoir logic


					if should_add:
						context_key = (b_idx, t_idx)
						full_feature_vector = feats_one_sae[b_idx, t_idx, :]
						
						#region** Storing nonzero vals and idxs
						
						top_vals, top_inds = torch.topk(full_feature_vector, k=min(MAX_STORED_FEATURES, full_feature_vector.shape[0]))
						valid_mask = top_vals > 0
						final_inds = top_inds[valid_mask].cpu()
						final_vals = top_vals[valid_mask].cpu()

						#endregion** Storing nonzero vals and idxs


						current_metadata = {
							'value': feats_one_sae[b_idx, t_idx, feat_idx].item(),
							'context': context_cache[context_key],
							'token_id': token_id,
							'step': batch_idx * current_batch_size + b_idx,
							'sparse_features': (final_inds, final_vals)
						}

						if slot_to_replace != -1:
							current_examples[slot_to_replace] = current_metadata
						else:
							current_examples.append(current_metadata)

		#endregion* Selection Logic


		#region* Saving

		if (batch_idx + 1) % CHECK_EVERY_N_BATCHES == 0:
			print(f"\n--- Checkpoint at Batch {batch_idx+1} ---")
			print("Saving current data chunk...")
			
			for i in range(num_saes):
				# Save the Dictionary for this SAE
				# Structure: {feat_idx: {'word_counts':..., 'examples':...}}
				data_to_save = feature_storage[i]
				
				chunk_save_path = f"{save_data_ps[i]}_batch_{batch_idx+1}.pkl"
				with open(chunk_save_path, "wb") as f:
					pickle.dump(data_to_save, f)
			print(f"Chunk saved to {chunk_save_path}")

			feature_storage = initialize_storage()
			gc.collect() 

		#endregion* Saving


		#region* Release memory

		del acts, feats_one_sae, current_acts, inputs_gpu, context_cache
		if 'all_found_indices' in locals(): del all_found_indices
		if 'combined_indices' in locals(): del combined_indices
		gc.collect() 
		torch.cuda.empty_cache()

		#endregion* Release memory


		loop.update(1)

#endregion Main Code





print("END FILE")