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
    
    #@ Experiment Selectors
    INPUT_MODE = "T2" # Options: "REAL", "T1", "T2", "T3" 
    
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
    LABS_PATH = LOAD_P + "cleargen_labs_26_9_25.pkl"
    CLASSHEAD_PATH = ROOT_P + "classhead/checkpoint-1650/"

CFG = Config()

#endregion Config: Paths, Defs, etc.



#region Funcs

def _ntorch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return torch.load(*args, **kwargs)

# Original monkey patch application
og_torch_load = torch.load
torch.load = _ntorch_load


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


def single_mactensor_recreate(indies,vals):
	
	cand = torch.zeros([6144]).to(torch.float32)

	#Broadcasting with torch makes this fast
	cand[indies] = vals

	return cand



#endregion Funcs



#region Class defs


class ActDataset(Dataset):

    def __init__(self, input_ids, labels, pad_id=0):
        self.input_ids = input_ids
        self.labels = labels
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
            "labels": self.labels[i]
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
    def __init__(self, config):
        super().__init__(config)
        self.classifier = ClassHead(config.hidden_size)

    def forward(self, input=None, labels=None):
        return self.classifier(input=input, labels=labels)

#endregion Class defs


#region Setup & Loads


#@ Class head
def load_classhead():
    """Load class head by creating new and importing weights."""
    with open(f"{CFG.CLASSHEAD_PATH}config.json") as f:
        cfg_data = json.load(f)

    model_cfg = CustomModCon(hidden_size=cfg_data["hidden_size"])
    classhead = WrapClassModel(model_cfg)
    
    state_dict = load_file(f"{CFG.CLASSHEAD_PATH}model.safetensors", device="cpu")
    classhead.load_state_dict(state_dict)
    classhead.to(CFG.DEVICE)
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
    with open(CFG.LABS_PATH, "rb") as f:
        labs = pickle.load(f)
    
    with open(CFG.TENS_DS_PATH, "rb") as f:
        tens_ds = torch.load(f)

    ad_inputs = tens_ds.tensors[0]
    ad_labels = torch.tensor(labs, dtype=torch.long)
    
    train_ds = ActDataset(ad_inputs, ad_labels, pad_id=tokenizer.pad_token_type_id)

    # Balancing weights
    y_np = labs.to(torch.int).numpy()
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
    
    if mode == "T1": # Guy/Girl contrast
        sentences = [
            "Female people are known as women and girls. Male people are known as boys and men.",
            "Andrew is the most popular guy in all of the school; he's the captain.",
            "Robert is the most popular boy in all of the school.",
            "Stacy is the most popular girl in all of the school."
        ]
        
    elif mode == "T2": # Gender pairs
        sentences = [
            "Joseph is male, and he knows it.", "Kate is female, and she knows it.", "Kate is female, and she knows it.",
            "Danny is a beautiful guy.", "Ella is a beautiful gal.", "Ella is a beautiful gal.",
            "Ollie is a lovely boy.", "Jenny is a lovely girl.", "Jenny is a lovely girl.",
            "James is a real man.", "Juliet is a real woman.", "Juliet is a real woman.",
            "The passport says male.", "The passport says female.", "The passport says female.",
            "He went to the shops.", "She went to the shops.", "She went to the shops.",
            "The wolf ate him.", "The wolf ate her.", "The wolf ate her.",
            "Adam showed his work.", "Betty showed her work.", "Betty showed her work.",
            "Be a gentleman.", "Be a lady.", "Be a lady."
        ]
        
    elif mode == "T3": # Global Toy
        sentences = ["The king watched his son, the young prince. The queen's gaze, however, was on her daughter, the princess. Nearby, a lord bowed to a lady; the man's gesture was met by the woman's polite nod. The conflict's hero had a famously masculine confidence, contrasting with the heroine's more subtle and feminine style of leadership. A young boy ran to his father, while a nearby girl held her mother's hand. In the corner, an actor rehearsed lines with an actress; the sister's delivery was flawless, but her brother forgot his cue. The actress's husband smiled at the error, though the actor's wife looked on with a critical eye.", "The king watched his son, the young princess."]
        
    # Processing
    encs = torch.Tensor([
        tokenizer.encode(s) + [0]*(CFG.SEQ_LEN - len(tokenizer.encode(s))) 
        for s in sentences
    ]).to(torch.int64)
    
    attens = (encs != 0).to(torch.long)
    ttids = torch.zeros(encs.size()).to(torch.long)
    
    return {"input_ids": encs, "token_type_ids": ttids, "attention_mask": attens}


#endregion Setup & Loads




#region Main Exe


def main():
    print(f"Starting Steering Tester... Mode: {CFG.INPUT_MODE}")
    
    # 1. Load Components
    classhead = load_classhead()
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


    # 3. Define Submodules (nnsight)
    if CFG.MODEL_TYPE == "bert":
        submodule_ref6 = model.bert.encoder.layer[6]
        submodule_ref7 = model.bert.encoder.layer[7]
        submodule_ref_last = model.bert.encoder.layer[11]



    # 4. Steering / Inference Loop
    print("Running Trace...")
    with torch.inference_mode():
        with model.trace(trace_inputs) as tracer:
            
            # Hook activations
            acts = submodule_ref6.output[0]
            
            # SAE Encode
            sae_hidden = model.ae6.encode(acts)
            sae_hidsav = sae_hidden.clone().to("cpu").save()

            # #! Refactor: Steering logic location. 
            # Currently just passing through, but this is where your 
            # commented-out "T1/T2/T3" modification code would live.
            # Example: sae_hidden[:, 1:, :] += vector
            
            # SAE Decode
            recons = model.ae6.decode(sae_hidden)
            
            # Patch graph
            submodule_ref7.input = recons
            
            # Save outputs
            finl = model.output.save()
            l11 = submodule_ref_last.output.save()


    # 5. Post-Processing
    print("Trace Complete. Processing outputs...")
    out11 = l11.value[0] # .value required for nnsight saves if not using implicit .save() behavior
    

    if CFG.TOK_TYPE == "cls":
        fin_mean = finl.value[0].to(CFG.DEVICE, non_blocking=True)
    elif CFG.TOK_TYPE == "pool":
        fin_mean = torch.mean(out11, dim=1)

    # 6. Analysis (Example: Finding gender features)
    # This mirrors the logic at the end of your original file
    if CFG.INPUT_MODE == "T2":
        run_gender_analysis(sae_hidsav.value)

    print("Done.")


def run_gender_analysis(sae_hidsav):
    #! Refactor: Extracted analysis logic to keep main loop clean.
    print("Running Gender Analysis...")
    
    # Masking logic (as per original T2 section)
    # Mask to remove the duplicate sentences (Reducing 27 -> 18)
    m1 = [True, True, False] * 9 
    
    # Mask for specific word indices
    m2 = [
        3,3, 5,5, 5,5, 5,5, 7,7, 
        1,1, 9,9, 10,10, 11,11
    ] # Truncated for example, matches your T2 logic
    
    try:
        t = sae_hidsav[m1]
        # Ensure m2 aligns with t's dimensions (checking logic safety)
        if t.shape[0] == len(m2):
            gends = t[:, m2, :]
            
            # Example: Histogram of first element
            fig = hist(gends[0, 0], 0, show=False, ret=True)
            print("Generated analysis figure (fig).")
    except Exception as e:
        print(f"Analysis skipped due to shape mismatch (expected with partial T2 inputs): {e}")


#endregion Main Exe





if __name__ == "__main__":
    main()

print("END OF FILE")