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
#* Main file for training SAEs.
#* Pretraining uses streamed uncopyrighted Pile, training on appox 640mil
#* Batch size = 128, Sequence Len = 512, Examples = 10_000
	#* Changed the examples to a larger number to see if we can get further improvements
#* Also does finetuning.

#? Has been optimised so it actually runs fairly quickly now
#endregion? Info 



#region Params

#region? SAE type

sae_type = "BTK"
# sae_type = "Standard"


#endregion? SAE type


#region? trn_stage
trn_stage = "pretrain"
# trn_stage = "finetune"
#endregion? trn_stage


#region? model_type
model_type = "bert"
# model_type = "bge"
#endregion? model_type


#region? tok_type
tok_type = "pool" #? Admittedly, we'll probably only be using pool, since CLS doesn't classify
# tok_type = "cls"
#endregion? tok_type


#region? f_c
f_c = 2 #Use only 2nd
# f_c = 3 #Use only 3rd
# f_c = 4 #Use second to last
# f_c = 5 #Use last
#endregion? f_c


#endregion Params



#region Functions and classes


class TrnStageException(Exception):

	def __init__(self,message = "No valid trn_stage set. Choose either 'finetune' or 'pretrain' "):
		super().__init__(self.message)
		self.message = message
	
	def __str__(self):
		return self.message


#region? StampLog

if (sae_type == "Standard"):
	#Defining an object to act as a timestamped log of all of the metrics
	StampedLog = namedtuple("StampedLog",["step","l0","l2","fve","mse","sparsity_loss","loss"])
elif (sae_type == "BTK"):
	StampedLog = namedtuple("StampedLog",["step","l0","l2","fve","auxk_loss","loss"])


#endregion? StampLog


#Used to create a data loader from the Asylex dataset
class AsyLexDataset(torch.utils.data.Dataset):

	def __init__(self, data):
		# super().__init__()
		self.data = data
	

	def __len__(self):
		return len(self.data)


	def __getitem__(self, i):
		return {"text":self.data[i]}



#region* Monkey patching torch.load (weights_only)

og_torch_load = torch.load

def ntorch_load(*args, **kwargs):
	kwargs['weights_only'] = False
	return og_torch_load(*args, **kwargs)

torch.load = ntorch_load #?its good practice to restore the function when done

#endregion


#endregion Functions and classes



#region Paths

root_p= "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/SAE_trainer/"
loads_p = root_p + "loads/"
saves_p = root_p + "saves/"


df_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/base_clean/our_train.pkl"


tens_ds_p = loads_p + f"{model_type}_tokds/gentpick{f_c}_tokds.pt"


#SAE locations
save_dirs = ["SAE3_saves/","SAE6_saves/","SAE9_saves/"]
log_save_dirs = ["SAE3_logs/","SAE6_logs/","SAE9_logs/"]


#endregion Paths



#region Vars


device = "cuda"
model_name = "bert-base-uncased" #! UNcased, UNCASED
seq_len = 512 #BERTs maximum sequence len
exp_factor = 8
act_dim = 768
lyndex = [3,6,9]



#region* When loading an SAE to finetune
if(trn_stage == "finetune"):
	pretrain_source_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/store/"
	pretrain_batch = 3 #Select which pretraining batch to take from
	pretrained_cp = [7000,7000,7000] #For all of SAE3/6/9, which checkpoint?
	pretrained_save_dirs = [pretrain_source_p+""+f"pred_SAEs/pre{pretrain_batch}_SAE_train_9-6-25/SAE{lyndex[i]}_saves/ae_{pretrained_cp[i]}.pt" for i in range(3)]
#endregion* When loading an SAE to finetune


if(trn_stage == "pretrain"):

	steps = 150_000 #curr Tested 10k,30k,60k,150k
	batch_size = 128
	step_size = 1000
	lstep_size = 200

	# save_steps = [1000,2000,3000,4000,5000,6000,7000,8000,9000,9999]
	save_steps = [step_size*i for i in range(1,steps//step_size+1)]
	save_steps[-1] -= 1

	log_steps = [lstep_size*i for i in range(1,steps//lstep_size+1)]
	log_steps[-1] = log_steps[-1] - 1 #Saving 1 before end


elif(trn_stage == "finetune"):
	steps = 180 #Derived as num_examples/batch = 23144/128 = 180.8125, with drop_last=True
	batch_size = 128 #Keeping the same as the pretrain
	
	save_steps = [45*i for i in range(1,5)]
	save_steps[-1] = save_steps[-1] - 1

	log_steps = [45*i for i in range(1,5)] #Tuned to save 4 times during the 180 steps
	log_steps[-1] = log_steps[-1] - 1 
	

else:
	raise TrnStageException()



#endregion Vars/Params



#region Model

model = LanguageModel(model_name,device_map=device)
# model.config.pad_token_id = tok.pad_token_type_id
model.eval()

#endregion Model



#region SAEs

trns = []


#region? trn_cfgs


if (sae_type == "Standard"):
	#? Standard SAE config
	trn_cfgs = [ dict(
				trainer=StandardTrainer,
				dict_class=AutoEncoder,
				activation_dim=act_dim,
				dict_size=exp_factor * act_dim ,
				lr=1e-4, #Update1 - at 1e-3 was irratic, had to reduce
				l1_penalty = 1e-3, #Default value 1e-1
				device=device,
				steps=steps,
				layer=lyndex[i],#? Whilst this is inaccurate, it only functions as metadata
				lm_name=model_name,
				warmup_steps=100, #About 1-10% of total steps as a heuristic. Try this first, see later
				sparsity_warmup_steps=100, #Shall assume the same holds for as for the above
				
	) for i in range(3)]


elif (sae_type == "BTK"):
	trn_cfgs = [ dict(
					trainer=BatchTopKTrainer, #Switched from Standard
					activation_dim=act_dim,
					dict_size=exp_factor * act_dim ,
					k = 350,
					#dict_class=AutoEncoder, #? Predefined in BTK
					# l1_penalty = 1e-1, #Default value 1e-1, also try 1e-4 #! Not in BTK
					# lr=1e-4, #Update1 - at 1e-3 was irratic, had to reduce
					lr = 1e-3, #Also try 1e-4
					device=device,
					steps=steps,
					layer=3,#? Whilst this is inaccurate, it only functions as metadata
					lm_name=model_name,
					warmup_steps=100, #About 1-10% of total steps as a heuristic. Try this first, see later
					# sparsity_warmup_steps=100, #! Not in BTK
					
	) for _ in range(3)]

#endregion? trn_cfgs


#region* Choose init


if (trn_stage == "pretrain"):

	#Defining and initialising trainers
	for init in trn_cfgs:
		trn = init.pop("trainer")
		trns.append(trn(**init))

#Load in the pre-existing state_dict.
elif(trn_stage == "finetune"):
	
	for i in range(len(pretrained_cp)):

		p = pretrained_save_dirs[i]

		with open(p,"rb") as f:
			
			trn = trn_cfgs[i].pop("trainer")
			trns.append(trn(**trn_cfgs[i]))
			precfg = torch.load(f)

			trns[-1].ae.load_state_dict(precfg)


else:
	raise TrnStageException()



#endregion* Choose init



#Modified to work with BERT instead - was structured like this originally to be able to specify submodule.
submodule_ref3 = eval(f"model.bert.encoder.layer[3]")
submodule_ref6 = eval(f"model.bert.encoder.layer[6]")
submodule_ref9 = eval(f"model.bert.encoder.layer[9]")


#endregion SAEs



#region Dataset and loader

#If pretraining, use uncopyrighted Pile
if trn_stage == "pretrain":

	dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		num_workers=4, #Start at this value, later we can move higher up - generally up to number of cores.
		prefetch_factor=8, #Decent to start here - how many batches are loaded at once
		persistent_workers=True,
	)




#region* Others


#? Realistically this should only be done once, but there isn't that much overhead so we can keep it like this for now.
#If finetuning, we use the Asylex dataset we preprocessed before.
elif trn_stage == "finetune":

	with open(df_p,"rb") as f:
		df = pickle.load(f)
		df_to_ds = list(df["raw_txt"])

	dataset = AsyLexDataset(df_to_ds)


	loader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			num_workers=4, #Start at this value, later we can move higher up - generally up to number of cores.
			prefetch_factor=8, #Decent to start here - how many batches are loaded at once
			persistent_workers=True,
			drop_last=True,
		)

else:
	raise TrnStageException()


#endregion* Others

#endregion Dataset and loader



#region Tokenizer
tok = model.tokenizer
tok.add_special_tokens({'pad_token': '[PAD]'})
tok.backend_tokenizer.enable_truncation(max_length=seq_len)
tok.backend_tokenizer.enable_padding(length=seq_len, pad_id=tok.pad_token_id,pad_token=tok.pad_token)
#endregion Tokenizer



#region Main Training Loop


#region* predefs
diter = iter(dataset)
step = 0
logs = [{} for _ in range(len(trns))]
#endregion* predefs


with tqdm(total=steps,desc="Training SAE...") as loop:
	
	while step < steps:
		

		#region* Inputs
		
		batch = next(diter)

		#? Needs to stay because of streamed data.
		inputs = tok(
			batch["text"],
			return_tensors="pt",
			padding="max_length",
			max_length=seq_len,
			truncation=True,
		)

		# inputs = {k: v.to("cpu", non_blocking=True) for k, v in inputs.items()}
		inputs = {k: v.to("cuda", non_blocking=True) for k, v in inputs.items()}

		#endregion* Inputs


		#region* Getting the activations from the model for this loop

		with torch.inference_mode():
			with model.trace(inputs, invoker_args={"max_length": seq_len}) as tracer:
				#Saving the activations from the different layers
				l3 = submodule_ref3.nns_output.save()
				l6 = submodule_ref6.nns_output.save()
				l9 = submodule_ref9.nns_output.save()
				

				# Have to call stop at layer 9 - works in terms of layer order, not code order
				submodule_ref9.nns_output.stop()

		#endregion* Getting the activations from the model for this loop



			#region* Saving as tensors
			#? Is under the inference mode from above

			#? These need to be called act
			act3 = l3[0].to(device,non_blocking=True)
			act3 = act3.view(-1,act3.size(-1))
			act6 = l6[0].to(device,non_blocking=True)
			act6 = act6.view(-1,act6.size(-1))
			act9 = l9[0].to(device,non_blocking=True)
			act9 = act9.view(-1,act9.size(-1))

			acts = [act3, act6, act9]


			#endregion* Saving as tensors


		#region* Updating the models, one at a time.
		

		for i in range(len(trns)):

			curr_act = acts[i]


			#region** Calculating Logs
			
			if (step in log_steps):
				
				with torch.no_grad():

					#Ensures that all the SAEs get their requisite input
					# curr_act = eval(f"act{lyndex[i]}") #? Now assigned at the start of the loop

					#Keeps the original on the other device,
					act_copy = curr_act.clone().to(trns[i].device)

					#region* Get Metrics

					#Losses contains l2, mse, spasity and recon losses.
					act_trn , recons, feats, losses = trns[i].loss(act_copy,step=step,logging = True,)

					l0 = (feats != 0 ).float().sum(dim=-1).mean().item()
					tot_var = torch.var(act_trn)
					res_var = torch.var(act_trn - recons)
					fve = 1 - (res_var/tot_var) #Fraction of variance explained

					#endregion* Get Metrics


					#region* StampLog
					
					
					if (sae_type == "Standard"):
						#Logging to a StampedLog
						logs[i][step] = StampedLog(
							step = step,
							l0 = l0,
							fve = fve.to("cpu").item(),
							l2 = losses["l2_loss"],
							mse = losses["mse_loss"],
							sparsity_loss = losses["sparsity_loss"],
							loss = losses["loss"],
						)
				

					elif (sae_type == "BTK"):
						logs[i][step] = StampedLog(
							step = step,
							l0 = l0,
							fve = fve.to("cpu").item(),
							l2 = losses["l2_loss"],
							auxk_loss = losses["auxk_loss"],
							loss = losses["loss"],
					)


					#endregion* StampLog

					#? Convoluted way of printing the last entry.
					print(logs[i][list(logs[i].keys())[-1]])

					#region*# Log saving! - Not here anymore
					
					# log_p = saves_p + f"{trn_stage}_SAEs/" + log_save_dirs[i]
					# log_s_p =  log_p + f"{trn_stage}_log_{step}.pkl"

					# if not (os.path.exists(log_p)):
					# 	os.mkdir(log_p)

					# #Saving progress every time in case of failure
					# with open(log_s_p,"wb") as f:
					# 	pickle.dump(logs[i],f)

					#endregion*# Log saving!


			#endregion** Logging stuff


			#region** Saving

			if (step in save_steps):


				#region* Saving checkpoint
				
				file_path = saves_p+ f"{trn_stage}_SAEs/" + save_dirs[i]
				namer = f"{trn_stage}ae_{step}.pt"


				if not (os.path.exists(file_path)):
					os.mkdir(file_path)
				
				#After the check has been done to make sure it exists.
				full_path = file_path + namer

				checkpoint = {k:v.cpu() for k,v in trns[i].ae.state_dict().items() }

				with open(full_path,"wb") as f:
					torch.save(checkpoint,f)

				#endregion* Saving checkpoint



				#region* Saving Logs
				
				log_p = saves_p + f"{trn_stage}_SAEs/" + log_save_dirs[i]
				# log_s_p =  log_p + f"{trn_stage}_log_{step}.pkl"
				log_s_p =  log_p + f"{trn_stage}_log_SAE{lyndex[i]}.pkl"


				if not (os.path.exists(log_p)):
					os.mkdir(log_p)
				


				# Save the ENTIRE logs[i] dictionary up to the current step
				with open(log_s_p,"wb") as f:
					pickle.dump(logs[i],f)



				#endregion* Saving Logs



			#endregion** Saving


			trns[i].update(step,acts[i])


		#endregion* Updating the models, one at a time.


		step+= 1
		loop.update(1)

#endregion Main Training Loop














