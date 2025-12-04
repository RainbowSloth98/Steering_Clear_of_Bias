import pandas as pd
import pickle
from tqdm import tqdm
import gc
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from nnsight import LanguageModel
from torch.utils.data import TensorDataset




#region# Combining and saving

base_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/saves/frag_saves/POOL/frag_v2/"

stack = []

for i in range(1,223):
	ld_p = base_p + f"cls_save_{i}.pt"

	with open(ld_p,"rb") as f:
		stack.append(pickle.load(f))

#TODO This is set to deal with odd batches. Fix to work with both
last = stack.pop(-1)
last = last.squeeze()

mrg = torch.cat(stack,dim=0)
mrg = mrg.reshape([-1,768])

s = torch.cat([mrg,last])


print("STOP")
print("STOP")



sp = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/saves/gendered/POOL/v_2/l11_gen_acts_v2.pt"

torch.save(s,sp)

print("STOP")
print("STOP")

#endregion# Combining and saving




#region Tuning


#region? frag_choice

#region? Choices
#* 1 = First chunk only
#* 2 = Second chunk only
#* 3 = Third chunk only; Where only 2 exist, take the latter.
#* 4 = prefinal/penultimate chunk only
#* 5 = final/ultimate chunk only
#endregion? Choices
frag_choice = 2

#endregion? frag_choice


#region? model_type


#region? Choices
#* bge = new model, should work better than bert
#* bert = just bert
#endregion? Choices

# model_type = "bge"
model_type = "bert"


if(model_type == "bert"):
	model_name = "bert-base-uncased"
elif(model_type == "bge"):
	model_name = "BAAI/bge-large-en-v1.5"


#endregion? model_type


#region? tok_type

#? Choices are POOL and CLS
tok_type = "POOL"

#endregion? tok_type


#region? act_select

# act_select = "middle"
act_select = "end"

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


#region? experiment_choice

#Experiment choice
# exp_choice = "og_clean"
exp_choice = "gender_clean"

#endregion? experiment_choice


#endregion Tuning



#region Params

#? Pre gender cleaning.
if(exp_choice == "og_clean"):
	total_examples = 23121
	batch_size=63 #divides 23121

#? Gender cleaned
elif (exp_choice == "gender_clean"):
	total_examples = 22299 
	batch_size=100 # divs 22299


device = "cuda"
seq_len = 512


run_for = total_examples//batch_size
if(total_examples%batch_size != 0):
	run_for += 1

save_steps = [i for i in range(1,run_for+1)]


#endregion Params



#region Paths

root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/"


#region* Loads

load_p = root_p +"loads/" 
tokds_lp = load_p + "tok_ds/"
# og_data_p = load_p + "our_train.pkl" #Functions as a new reference table.


#* Example path: loads/tok_ds/bert_gendered/gentpick3_tokds.pt
cls_lp = tokds_lp + f"{model_type}_gendered/gentpick{frag_choice}_tokds.pt"


#endregion* Loads


#region* Saves

if (model_type == "bge"): #! BGE has its own data file
	save_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/bge-large-en/saves/" 
elif(model_type == "bert"):
	save_p = root_p + "saves/"


#? Frag saves are all stored here, regardless of experiment
frag_sp = save_p + "frag_saves/"
cls_sp = frag_sp + "CLS/" #TODO Remove this. CLS doesn't work
pool_sp = frag_sp + "POOL/"


#* Example path: saves/frag_saves/POOL/frag_v2/
tosave_p = frag_sp + f"{tok_type}/frag_v{frag_choice}/"


#endregion* Saves



#endregion Paths



#region Model and submodules

model = LanguageModel(model_name,device_map=device)
model.eval() #No changing weights.


sub = eval(f"model.bert.encoder.layer[{act_type}]")



#endregion Model and submodules



#region Tokenizer

tok = model.tokenizer
tok.add_special_tokens({'pad_token': '[PAD]'})
tok.backend_tokenizer.enable_truncation(max_length=seq_len)
tok.backend_tokenizer.enable_padding(length=seq_len, pad_id=tok.pad_token_id,pad_token=tok.pad_token)

#endregion Tokenizer



#region Dataset loading

# #?Loading a TensorDataset version of our data
# with open(lab_chunk_ds_p,"rb") as f:
# 	lab_chunk_ds = torch.load(f)

# Loading the first only chunked stuff
with open(cls_lp,"rb") as f:
	tokds = torch.load(f,weights_only=False)


# #Original data - needed in order to find indicies of where we extracted
# with open(og_data_p,"rb") as f:
# 	og_data= pickle.load(f)


loader = torch.utils.data.DataLoader(
	# lab_chunk_ds, #? Switching to test first only
	tokds,
	batch_size=batch_size,
	num_workers=4, #Start at this value, later we can move higher up - generally up to number of cores.
	prefetch_factor=8, #Decent to start here - how many batches are loaded at once
	persistent_workers=True,

)




#endregion Dataset loading




#region Main Loop

diter = iter(loader)
saved_acts = []


with tqdm(total=run_for, desc="Getting Acts...") as loop:
	for batch_idx in range(run_for):


		# exs, _ = next(diter) #? If the dataset has labels, but we don't use them
		exs = next(diter) #? if the dataset has no labels.3

		exs = [t.to(device, non_blocking=True) for t in exs]

		#region* Extracting activations

		with torch.inference_mode():
			with model.trace(*exs) as tracer:

					l11 = sub.nns_output.save() #TODO Name is OOD

		#endregion* Extracting activations



		#region* Processing extracted acts

		if(tok_type == "CLS"):
			saved_acts.append(l11[0][:,0,:].to("cpu",non_blocking=True)) #[batch,seq,embed]/[63,512,768]; [63,512,1024] for bge
		
		elif(tok_type == "POOL"):
			saved_acts.append(torch.mean(l11[0],dim=1).to("cpu"))

		#endregion* Processing extracted acts



		#region* Saving!

		if loop.n in save_steps:
			# to_save = l11[0].to("cpu",non_blocking=True)
			sp = tosave_p + f"cls_save_{loop.n}.pt"
			to_save = torch.stack(saved_acts)

			with open(sp,"wb") as f:
				pickle.dump(to_save,f)


			del to_save

			while saved_acts:
				del saved_acts[0]
			gc.collect()


		#endregion* Saving!




		loop.update(1)

#endregion Main Loop






print("End File")


#region Unused


#region# Extracting full tokenised dataset (from gendered data from 30.9.25), done on 1.10.25

# gendat_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/loads/cleargen_train_26_9_25.pkl"

# tokds_sp = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/loads/tok_ds/bert_gendered/tpickfull.pkl"

# with open(gendat_p,"rb") as f:
# 	gendat = pickle.load(f)

# raws = list(gendat["raw_txt"])

# toks = []

# for doc in tqdm(raws,desc="Processing documents..."):
# 	toks.append(chunk_document(doc,tok))


# tpicklens = [len(doc) for doc in toks]



# with open(tokds_sp,"wb") as f:
# 	pickle.dump(toks,f)

# tpicklens_sp = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/loads/tok_ds/bert_gendered/tpicklens.pkl"

# with open(tpicklens_sp,"wb") as f:
# 	pickle.dump(tpicklens,f)



# print("STOP")
# print("STOP")

#endregion# Extracting full tokenised dataset (from gendered data from 30.9.25), done on 1.10.25



#region# Creating tokenised subdata set p2. (from the gendered data from 30.9.25)

# gen_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/store/genderer/26-9-25/"
# lab_p = gen_p + "cleargen_labs_26_9_25.pkl"

# out_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/loads/tok_ds/bert_gendered/"
# tpickfull_p = out_p + "tpickfull.pkl"


# with open(lab_p,"rb") as f:
# 	lab = pickle.load(f)

# with open(tpickfull_p,"rb") as f:
# 	toks = pickle.load(f)

# f_c = 5

# # * Deciding how to sample chunks
# if (f_c == 2):
# 	tpick = [t[1] for t in toks] #Take the second
# elif (f_c == 3):
# 	tpick = [t[2] if t.size()[0] >=3 else t[-1] for t in toks] #Take third if exists
# elif (f_c == 4):
# 	tpick = [t[-2] for t in toks] #Take second to last
# elif (f_c == 5):
# 	tpick = [t[-1] for t in toks] #Take last

# tens = torch.stack(tpick)

# att = (tens != 0).to(torch.int64)

# tokds = TensorDataset(tens,att)


# o_p = out_p + f"tpick{f_c}_tokds.pt"
# with open(o_p,"wb") as f:
# 	torch.save(tokds,f)



#endregion# Creating tokenised subdata set p2. (from the gendered data from 30.9.25)



#region# Redid dataset to add the attention mask.

# t_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/first_only/first_only_tensor.pt"

# with open(t_p,"rb") as f:
# 	t = torch.load(f)

# att_mask = torch.full(t.size(),1,dtype=torch.int64)


# fo_chunks_ds = TensorDataset(t,att_mask)


# stp = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/first_only/first_only_ds.pt"
# with open(stp,"wb") as f:
# 	torch.save(fo_chunks_ds,f)

# print("STOP")
# print("STOP")

#endregion# Redid the ds to add the attention mask.



#region# Funcs

# #Take a long document, produce a tensor of its overlapped chunks which start with a CLS, and end on a SEP.
# def chunk_document(document: str, tokenizer, max_len: int = 512, overlap: int = 256):
# 	"""
# 	Chunks a long document into overlapping segments suitable for BERT-based models.

# 	This function tokenizes the document, splits it into chunks of a specified
# 	maximum length with a given overlap, and formats each chunk with special
# 	tokens ([CLS], [SEP]) and padding.

# 	Args:
# 		document (str): The long text document to be chunked.
# 		tokenizer: An instance of a Hugging Face tokenizer (e.g., BertTokenizer).
# 		max_len (int): The maximum sequence length for each chunk (e.g., 512 for BERT).
# 		overlap (int): The number of tokens to overlap between consecutive chunks.

# 	Returns:
# 		torch.Tensor: A tensor of shape [num_chunks, max_len] containing the
# 					tokenized and formatted chunks. Returns an empty tensor
# 					if the document is empty.
# 	"""
# 	# 1. Tokenize the entire document without adding special tokens yet.
# 	# We get a list of token IDs.
# 	token_ids = tokenizer.encode(document, add_special_tokens=False)

# 	# If the document is empty or has no tokens, return an empty tensor.
# 	if not token_ids:
# 		return torch.empty((0, max_len), dtype=torch.long)

# 	# 2. Define the size of the content window for each chunk.
# 	# This is max_len minus 2 to account for [CLS] and [SEP] tokens.
# 	content_window_size = max_len - 2

# 	# 3. Calculate the stride, which is how many tokens to jump for the next chunk.
# 	stride = content_window_size - overlap
# 	if stride <= 0:
# 		raise ValueError(
# 			f"Overlap ({overlap}) must be smaller than the content window size "
# 			f"({content_window_size}) to avoid infinite loops or redundant chunks."
# 		)

# 	# 4. Get special token IDs from the tokenizer.
# 	cls_token_id = tokenizer.cls_token_id
# 	sep_token_id = tokenizer.sep_token_id
# 	pad_token_id = tokenizer.pad_token_id

# 	chunked_token_ids = []

# 	# 5. Iterate through the token_ids to create chunks.
# 	for i in range(0, len(token_ids), stride):
# 		# Extract the content for the current chunk.
# 		chunk_content = token_ids[i : i + content_window_size]

# 		# If the extracted chunk is empty (can happen at the very end), skip it.
# 		if not chunk_content:
# 			continue

# 		# 6. Format the chunk with special tokens.
# 		# [CLS] + chunk_content + [SEP]
# 		formatted_chunk = [cls_token_id] + chunk_content + [sep_token_id]

# 		# 7. Pad the chunk to max_len if it's shorter.
# 		padding_needed = max_len - len(formatted_chunk)
# 		padded_chunk = formatted_chunk + ([pad_token_id] * padding_needed)

# 		# Add the processed chunk to our list.
# 		chunked_token_ids.append(padded_chunk)

# 	# 8. Convert the list of lists into a PyTorch tensor.
# 	return torch.tensor(chunked_token_ids, dtype=torch.long)


#region*# monkey patching torch.load

# og_torch_load = torch.load

# def ntorch_load(*args, **kwargs):
# 	kwargs['weights_only'] = False
# 	return og_torch_load(*args, **kwargs)

# torch.load = ntorch_load #?its good practice to restore the function when done

#endregion* monkey patching torch.load

#endregion# Funcs



#region# Classes


# class ActDataset(Dataset):

#     def __init__(self, actensors, label):
#         # super().__init__()
#         self.actensors = actensors
#         self.label = label
    

#     def __len__(self):
#         return len(self.label)


#     def __getitem__(self, i):
#         return {"input":self.actensors[i],"label":self.label[i]}


#endregion# Classes



#endregion Unused

