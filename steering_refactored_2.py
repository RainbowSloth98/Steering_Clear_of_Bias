import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Third-party / Specific
from tqdm import tqdm
from functools import partial
from collections import Counter
from safetensors.torch import load_file
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from transformers import PreTrainedModel, PretrainedConfig

# NNsight & Dictionary Learning
from nnsight import LanguageModel
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE


#region? Info:

#* Steering file for only BTK style SAEs.
#* T1: Only male and female feature
#* T2: Above, but with exclusive co-activating features too.
    #? Need to have an idea of which value to set them to.


#endregion? Info:



#region Setup & Loads

def load_traindf (path):
    path = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/steerer/loads/cleargen_labs_26_9_25.pkl"
    
    with open(path,"rb") as f:
        labs= pickle.load(f)

#@ Class head
def load_classhead(classhead_p,device):
    """Load class head by creating new and importing weights."""
    with open(f"{classhead_p}config.json") as f:
        cfg_data = json.load(f)

    model_cfg = CustomModCon(hidden_size=cfg_data["hidden_size"])
    classhead = WrapClassModel(model_cfg)
    
    state_dict = load_file(f"{classhead_p}model.safetensors", device="cpu")
    classhead.load_state_dict(state_dict)
    classhead.to(device)
    classhead.eval()
    return classhead


#@ Model
def setup_nnsight_model():
    """Initializes the NNsight LanguageModel and SAE."""
    model = LanguageModel(CFG.MODEL_NAME, device_map=CFG.DEVICE)
    model.eval()
    
    # Tokenizer setup
    tok = model.tokenizer
    tok.add_special_tokens({'pad_token': '[PAD]'})
    tok.backend_tokenizer.enable_truncation(max_length=CFG.SEQ_LEN)
    tok.backend_tokenizer.enable_padding(length=CFG.SEQ_LEN, pad_id=tok.pad_token_id, pad_token=tok.pad_token)
    
    # SAE Load
    ae6 = BatchTopKSAE.from_pretrained(CFG.SAE_PATH, device=CFG.DEVICE)
    model.ae6 = ae6 # Register SAE
    
    return model, tok



#@ Data Loader
def get_dataloader(tokenizer):
    """Loads datasets, computes weights, and returns the loader."""
    # 1. Load Task Labels
    with open(CFG.TASKLABS_PATH, "rb") as f:
        tasklabs = pickle.load(f)
    
    # 2. Load Gender Labels
    # Using weights_only=False as standard torch objects (lists/tensors) often require it
    genderlabs = torch.load(CFG.GENDLABS_PATH, weights_only=False)
    
    # 3. Load Inputs
    with open(CFG.TENS_DS_PATH, "rb") as f:
        tens_ds = torch.load(f, weights_only=False)

    ad_inputs = tens_ds.tensors[0]
    ad_labels = torch.tensor(tasklabs, dtype=torch.long)
    
    # Check format of genderlabs and convert to tensor
    if isinstance(genderlabs, list):
        ad_gender_labels = torch.tensor(genderlabs, dtype=torch.long)
    else:
        ad_gender_labels = genderlabs.to(dtype=torch.long)
    
    # Pass both label sets to dataset
    train_ds = ActDataset(ad_inputs, ad_labels, ad_gender_labels, pad_id=tokenizer.pad_token_type_id)

    # Balancing weights (Based on Task Labels)
    y_np = tasklabs.to(torch.int).numpy()
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_np), y=y_np
    )
    weight_map = dict(zip(np.unique(y_np), class_weights))
    samples_weight = torch.tensor([weight_map[x] for x in y_np], dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=samples_weight, num_samples=len(samples_weight), replacement=True
    )

    loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler, 
        shuffle=False, num_workers=4, prefetch_factor=8, persistent_workers=True
    )
    return loader


#@ Inputs
def get_input_sentences(mode, tokenizer):
    sentences = []
    
        
    if mode == "Tn":
        
        sentences = [
            "Joseph is male, and he knows it.", 
            "Kate is female, and she knows it.", 
            "Kate is female, and she knows it.",
            "Danny is a beautiful guy.", 
            "Ella is a beautiful gal.", 
            "Ella is a beautiful gal.",
            "Ollie is a lovely boy.", 
            "Jenny is a lovely girl.", 
            "Jenny is a lovely girl.",
            "James is a real man.", 
            "Juliet is a real woman.", 
            "Juliet is a real woman.",
            "The passport says male.", 
            "The passport says female.", 
            "The passport says female.",
            "He went to the shops.", 
            "She went to the shops.", 
            "She went to the shops.",
            "The wolf ate him.", 
            "The wolf ate her.", 
            "The wolf ate her.",
            "Adam showed his work.", 
            "Betty showed her work.", 
            "Betty showed her work.",
            "Be a gentleman.", 
            "Be a lady.", 
            "Be a lady."
        ]
        

    elif mode == "Tx": # Example of other
        pass



    # Processing
    encs = torch.Tensor([
        tokenizer.encode(s) + [0]*(CFG.SEQ_LEN - len(tokenizer.encode(s))) 
        for s in sentences
    ]).to(torch.int64)
    
    attens = (encs != 0).to(torch.long)
    ttids = torch.zeros(encs.size()).to(torch.long)
    
    return {"input_ids": encs, "token_type_ids": ttids, "attention_mask": attens}


#endregion Setup & Loads



#region Funcs


def collect_difference_vectors(model, sae, paired_loader, device="cuda"):
	"""
	Phase 1: Processes counterfactual pairs to extract feature-wise differences.
	
	Args:
		model: The NNsight language model.
		sae: The loaded SAE.
		paired_loader: A DataLoader yielding batches of paired inputs 
					(e.g., {'original': batch_a, 'counterfactual': batch_b}).
		device: 'cuda' or 'cpu'.
		
	Returns:
		torch.Tensor: A tensor of shape [N_samples, N_features] containing
					(Act_counterfactual - Act_original).
	"""
	diff_vectors = []
	
	print("Phase 1: Collecting Difference Vectors...")
	
	# Disable gradients for inference speed
	with torch.no_grad():
		for batch in tqdm(paired_loader, desc="Processing Pairs"):
			
			# 1. Unpack batch
			# Assuming loader returns dict with 'original' and 'counterfactual' keys
			# containing tokenized inputs.
			input_og = batch['original'].to(device)
			input_cf = batch['counterfactual'].to(device)
			
			# 2. Trace Original
			# We use a clean context for each to avoid graph leakage
			with model.trace(input_og) as tracer:
				# Hook point depends on your specific model setup (e.g., layer 6 output)
				acts_og = model.bert.encoder.layer[6].output[0] 
				sae_latents_og = sae.encode(acts_og)
				sae_latents_og.save()
				
			# 3. Trace Counterfactual
			with model.trace(input_cf) as tracer:
				acts_cf = model.bert.encoder.layer[6].output[0]
				sae_latents_cf = sae.encode(acts_cf)
				sae_latents_cf.save()
			
			# 4. Compute Difference (Delta)
			# We usually average over the sequence length to get one vector per example
			# Or you can target specific token positions if known.
			# Here, we use mean pooling over the sequence for robustness.
			vec_og = torch.mean(sae_latents_og.value, dim=1) 
			vec_cf = torch.mean(sae_latents_cf.value, dim=1)
			
			# Delta: Direction FROM Original TO Counterfactual
			# Positive values = Features added in CF
			# Negative values = Features removed in CF
			delta = vec_cf - vec_og
			
			diff_vectors.append(delta.cpu())

	# Stack into single tensor: [Total_Samples, Feature_Dim]
	full_diff_tensor = torch.vstack(diff_vectors)
	return full_diff_tensor



def hist(tens: torch.Tensor, idx: int, show=False, ret=True):
    """Creates a Plotly bar chart of SAE activations."""
    data = tens.detach().cpu().numpy()
    colors = np.array(['#E2E2E2'] * len(data)) 
    
    masked_data = data.copy()
    masked_data[idx] = -np.inf
    top_5_indices = np.argsort(masked_data)[-5:]
    
    colors[top_5_indices] = '#636EFA' 
    colors[idx] = '#EF553B' 

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=np.arange(len(data)), y=data,
        marker_color=colors, marker_line_width=0,
        hoverinfo='x+y', name='Activations'
    ))

    fig.update_layout(
        title=f"SAE Feature Activations (Target: {idx})",
        xaxis_title="Feature Index", yaxis_title="Activation Value",
        template="plotly_white", bargap=0, showlegend=False
    )

    if show: fig.show()
    if ret: return fig



def show_act(tens, namer):
    """Simple line plot for tensor values."""
    fig = go.Figure(data=go.Scatter(y=tens, mode='lines', name='Tensor Values'))
    fig.update_layout(title=namer, xaxis_title='Index', yaxis_title='Value')
    return fig



def create_overlapping_chunks(tokens, chunk_size=512, overlap=100):
	"""Yields successive overlapping chunks from a list of tokens."""
	if not tokens:
		return
	
	# The step size is the chunk size minus the overlap
	step = chunk_size - overlap
	for i in range(0, len(tokens), step):
		yield tokens[i:i + chunk_size]



def recreate_feat_dist(sf_inp: tuple):
	idxs,vals = sf_inp
	ret = torch.zeros([6144])
	ret[idxs] = vals
	return ret




#endregion Funcs



#region Class defs


class ActDataset(Dataset):

    def __init__(self, input_ids, labels, gender_labels, pad_id=0):
        self.input_ids = input_ids
        self.labels = labels
        self.gender_labels = gender_labels # NEW: Store gender labels
        self.pad_id = pad_id


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, i):
        input_id = self.input_ids[i]
        attention_mask = (input_id != self.pad_id).long()
        token_type_ids = torch.zeros_like(input_id)

        return {
            "input_ids": input_id,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": self.labels[i],
            "gender_labels": self.gender_labels[i] # NEW: Return paired gender label
        }


class ClassHead(nn.Module):
    def __init__(self, hidden_d):
        super().__init__()
        self.classifier = nn.Linear(hidden_d, 1)

    def forward(self, input=None, labels=None):
        logits = self.classifier(input)
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.view(-1), labels.float()) #Labels must be float: BCEWithLogitsLoss
        return {"logits": logits, "loss": loss}


class CustomModCon(PretrainedConfig):
    def __init__(self, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class WrapClassModel(PreTrainedModel):
    config_class = CustomModCon # explicit link to config

    def __init__(self, config):
        super().__init__(config)
        self.classifier = ClassHead(config.hidden_size)

    def forward(self, input=None, labels=None):
        return self.classifier(input=input, labels=labels)

CustomModCon.register_for_auto_class()

#endregion Class defs




#region Config: Paths, Defs, etc.


class Config:
    """Paths, defs, params and others"""
    # Model Params
    MODEL_TYPE = "bert" # Options: "bert", "bge"
    TOK_TYPE = "pool"
    LAYER_IDX = 6      # Steering layer
    
    # Run Params
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEQ_LEN = 512
    BATCH_SIZE = 70
    DATA_SIZE = 22299
    
    #@ Configs
    INPUT_MODE = "REAL" # Options: "REAL" / Tx, where x=1,2,3...
    # INPUT_MODE = "Tn" # Options: "REAL" / Tx, where x=1,2,3...
    # INPUT_MODE = "Tx" # Options: "REAL" / Tx, where x=1,2,3...
    
    MFEAT = 4364
    FFEAT = 1562
    MFCHANG = 3.75

    # Paths
    ROOT_P = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/steerer/"
    LOAD_P = ROOT_P + "loads/"
    SAVE_P = ROOT_P + "saves/"
    STORE_P = "/media/strah/344A08F64A08B720/Work_related/store/"
    
    # Dynamic Paths (Properties)
    @property
    def MODEL_NAME(self):
        return "bert-base-uncased" if self.MODEL_TYPE == "bert" else "BAAI/bge-large-en-v1.5"
    
    @property
    def ACT_DIM(self):
        return 768 if self.MODEL_TYPE == "bert" else 1024

    @property
    def TENS_DS_PATH(self):
        # Assuming f_c was hardcoded to 2 in original, logic can be added here if f_c varies
        return self.LOAD_P + f"{self.MODEL_TYPE}_tokds/gentpick2_tokds.pt"

    SAE_PATH = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/mact/loads/BTK_SAE6_144000.pt"
    TASKLABS_PATH = LOAD_P + "cleargen_labs_26_9_25.pkl"
    GENDLABS_PATH = LOAD_P + "real_malefem_inlp_labs_7_11_25.pt"
    CLASSHEAD_PATH = ROOT_P + "classhead/checkpoint-1650/"
    GENDERHEAD_PATH = ROOT_P + "genderhead/l11_checkpoint-1263/"

    classhead = load_classhead(CLASSHEAD_PATH,DEVICE)
    genderhead = load_classhead(GENDERHEAD_PATH,DEVICE)

    classhead.eval()
    genderhead.eval()



CFG = Config()

#endregion Config: Paths, Defs, etc.







#region Main Exe



#@ Steer defs
#? Consider featurs 1043 for main person, and 1186 for entourage.
#Change nothing
def steering_tn(sae_hidden):

    og_sav = sae_hidden.clone().to("cpu").save()

    sae_hidden[0,1,CFG.MFEAT] = sae_hidden[0,1,CFG.MFEAT] / CFG.MFCHANG
    sae_hidden[0,1,4889] = sae_hidden[0,1,4889] / 2
    sae_hidden[0,1,6124] = sae_hidden[0,1,6124] / 2

    sae_hidden[0,1,CFG.FFEAT] = sae_hidden[0,1,CFG.FFEAT] * (1.5 * CFG.MFCHANG)
    sae_hidden[0,1,1914] = sae_hidden[0,1,1914] * 2


    n_sav = sae_hidden.clone().to("cpu").save()

    return n_sav,og_sav


#Example to be copied
def steering_tx(sae_hidden):


    og_sav = sae_hidden.clone().to("cpu").save()
    n_sav = sae_hidden.clone().to("cpu").save()

    return n_sav,og_sav


#curr Testing Steering funcs
def steering_t0(sae_hidden):

    sae_hidden[:]

    return sae_hidden.clone().to("cpu").save()





def main():
    print(f"Starting Steering Tester... Mode: {CFG.INPUT_MODE}")
    
    # 1. Load Components
    model, tok = setup_nnsight_model()
    loader = get_dataloader(tok)

    diffs = collect_difference_vectors(model,model.ae6,loader)





def steer_main():
    print(f"Starting Steering Tester... Mode: {CFG.INPUT_MODE}")
    
    # 1. Load Components
    model, tok = setup_nnsight_model()
    loader = get_dataloader(tok)
    

    # 2. Select Inputs
    if CFG.INPUT_MODE == "REAL":
        diter = iter(loader)
        batch = next(diter)
        # Reconstruct batch for tracing
        token_type_ids = torch.zeros(batch['input_ids'].size())
        trace_inputs = {
            "input_ids": batch['input_ids'].type(torch.int64).to(CFG.DEVICE),
            "token_type_ids": token_type_ids,
            "attention_mask": batch['attention_mask']
        }
    else: #Tests
        trace_inputs = get_input_sentences(CFG.INPUT_MODE, tok)
        trace_inputs = {k: v.to(CFG.DEVICE) for k, v in trace_inputs.items()}
        # For tests, we might not have a real batch, so we pass None to analysis or handle it there
        batch = None 



    # 3. Define Submodules (nnsight)
    if CFG.MODEL_TYPE == "bert":
        submodule_ref6 = model.bert.encoder.layer[6]
        submodule_ref7 = model.bert.encoder.layer[7]
        submodule_ref_last = model.bert.encoder.layer[11]



    # 4. Steering / Inference Loop
    print("Running Trace...")
    with torch.inference_mode():
        with model.trace(trace_inputs) as tracer:
            
            
            acts = submodule_ref6.nns_output[0] #Hook acts
            sae_hidden = model.ae6.encode(acts) # SAE Encode



            #STEERING HERE
            # n_sav,og_sav = steering_tn(sae_hidden) #@ Intervention
            n_sav,og_sav = steering_tx(sae_hidden)



            recons = model.ae6.decode(sae_hidden)# SAE Decode
            submodule_ref7.input = recons # Patch graph
            

            finl = model.output.save() # Save outputs
            l11 = submodule_ref_last.nns_output.save()



    # 5. Post-Processing
    print("Trace Complete. Processing outputs...")
    pre_mean = l11[0] # .value required for nnsight saves if not using implicit .save() behavior
    fin_mean = torch.mean(pre_mean, dim=1) #? We don't use the CLS method anymore.


    # 6. Analysis (Example: Finding gender features)
    # This mirrors the logic at the end of your original file
    #@ Analysis select
    if(CFG.INPUT_MODE == "Tn"):
        tn_analysis(fin_mean,pre_mean,n_sav,og_sav,model,tok)
    
    elif CFG.INPUT_MODE == "T0":
        t0_analysis(fin_mean,pre_mean,n_sav,og_sav)
    
    elif CFG.INPUT_MODE == "REAL":
        real_analysis(fin_mean,pre_mean,n_sav,og_sav,tok,batch)

    print("Done.")







#@ Analysis def
#TODO
def real_analysis(fin_mean, pre_mean, n_sav, og_sav, tok, inp_batch):



    print("STOP")
    print("STOP")

    # 1. Get Predictions from Heads
    classed_out = CFG.classhead(fin_mean)
    gended_out = CFG.genderhead(fin_mean)
    
    print("STOP")
    print("STOP")

    # 2. Get Ground Truths from the Batch
    # inp_batch comes from the DataLoader/ActDataset and contains the paired labels
    gt_task = inp_batch["labels"].cpu()
    gt_gender = inp_batch["gender_labels"].cpu()


    print("STOP")
    print("STOP")

    # 3. Process Preds (Logits -> Binary)
    # Assuming standard logits where >0 is class 1
    pred_task = (classed_out["logits"] > 0).long().cpu().flatten()
    pred_gender = (gended_out["logits"] > 0).long().cpu().flatten()

    print("STOP")
    print("STOP")

    # 4. Display Comparison
    print("\n--- Analysis Report ---")
    print(f"Batch Size: {len(gt_task)}")
    
    # Simple accuracy check
    task_acc = (pred_task == gt_task).float().mean().item()
    gender_acc = (pred_gender == gt_gender).float().mean().item()
    
    print(f"Task Accuracy: {task_acc:.2%}")
    print(f"Gender Accuracy: {gender_acc:.2%}")
    
    # Show first 10 examples for manual inspection
    print("\nFirst 10 Examples:")
    print(f"{'GT Task':<10} | {'Pred Task':<10} | {'GT Gender':<10} | {'Pred Gender':<10}")
    print("-" * 50)
    for i in range(min(10, len(gt_task))):
        print(f"{gt_task[i].item():<10} | {pred_task[i].item():<10} | {gt_gender[i].item():<10} | {pred_gender[i].item():<10}")

    print("STOP")
    print("STOP")


def tn_analysis(fin_mean,pre_mean,n_sav,og_sav,model,tok):
    
    with torch.no_grad():
        pred_scores = model.cls(pre_mean[0])
    
    pred_ids = torch.argmax(pred_scores,dim=-1)

    dec = tok.decode(pred_ids)


    #TODO See what the class and gender heads say about the steering


    print("STOP")
    print("STOP")



def t0_analysis(fin_mean,pre_mean,n_sav,og_sav):
    pass



#endregion Main Exe





if __name__ == "__main__":
    main()

print("END OF FILE")
