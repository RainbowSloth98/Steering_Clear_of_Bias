import os
import json
import pickle
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
from tqdm import tqdm
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans
from transformers import PreTrainedModel, PretrainedConfig

# NNsight & Dictionary Learning
from nnsight import LanguageModel
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE



#region Config: Paths, Defs, etc.

class Config:
    """Paths, defs, params and others"""
    # Model Params
    MODEL_TYPE = "bert" # Options: "bert", "bge"
    LAYER_IDX = 6      # Steering layer
    
    # Run Params
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEQ_LEN = 512
    BATCH_SIZE = 32 # Adjusted for inference
    
    # Steering Configs
    MFEAT = 4364
    FFEAT = 1562
    
    # Paths (Update these absolute paths as needed)
    ROOT_P = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/steerer/"
    LOAD_P = ROOT_P + "loads/"
    
    # Data Paths
    SAE_PATH = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/mact/loads/BTK_SAE6_144000.pt"
    GENDER_BEND_DATA = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/testpy/saves/small_genderbends.pkl"
    
    # Eval Data Paths
    TENS_DS_PATH = LOAD_P + f"{MODEL_TYPE}_tokds/gentpick2_tokds.pt"
    TASKLABS_PATH = LOAD_P + "cleargen_labs_26_9_25.pkl"
    GENDLABS_PATH = LOAD_P + "real_malefem_inlp_labs_7_11_25.pt"
    
    # Head Paths
    CLASSHEAD_PATH = ROOT_P + "classhead/checkpoint-1650/"
    GENDERHEAD_PATH = ROOT_P + "genderhead/l11_checkpoint-1263/"

    @property
    def MODEL_NAME(self):
        return "bert-base-uncased" if self.MODEL_TYPE == "bert" else "BAAI/bge-large-en-v1.5"

#endregion Config


#region Class/Head Definitions (For Analysis)

class ClassHead(nn.Module):
    def __init__(self, hidden_d):
        super().__init__()
        self.classifier = nn.Linear(hidden_d, 1)

    def forward(self, input=None, labels=None):
        logits = self.classifier(input)
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.view(-1), labels.float())
        return {"logits": logits, "loss": loss}


class CustomModCon(PretrainedConfig):
    def __init__(self, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class WrapClassModel(PreTrainedModel):
    config_class = CustomModCon
    def __init__(self, config):
        super().__init__(config)
        self.classifier = ClassHead(config.hidden_size)
    def forward(self, input=None, labels=None):
        return self.classifier(input=input, labels=labels)

CustomModCon.register_for_auto_class()

def load_classhead(classhead_p, device):
    """Load class head by creating new and importing weights."""
    try:
        with open(f"{classhead_p}config.json") as f:
            cfg_data = json.load(f)

        model_cfg = CustomModCon(hidden_size=cfg_data["hidden_size"])
        classhead = WrapClassModel(model_cfg)
        
        state_dict = load_file(f"{classhead_p}model.safetensors", device="cpu")
        classhead.load_state_dict(state_dict)
        classhead.to(device)
        classhead.eval()
        return classhead
    except Exception as e:
        print(f"Warning: Failed to load classhead from {classhead_p}. Error: {e}")
        return None

#endregion Class/Head Definitions


#region Data Processing Helper (Collection Phase)

class PairedDataset(Dataset):
    """ Dataset that returns paired tokenized inputs (Original vs Counterfactual). """
    def __init__(self, df, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.originals = df['raw_txt'].tolist()
        self.counterfactuals = df['genderbent_text'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        og_enc = self.tokenizer(self.originals[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        cf_enc = self.tokenizer(self.counterfactuals[idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'original': og_enc['input_ids'].squeeze(0),
            'counterfactual': cf_enc['input_ids'].squeeze(0)
        }

def prepare_paired_dataloader(df, tokenizer, batch_size=32):
    dataset = PairedDataset(df, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def load_og_and_coef_data(path):
    with open(path,"rb") as f:
        df = pickle.load(f)
    return df

#endregion Data Processing Helper


#region Data Processing Helper (Eval Phase)


class ActDataset(Dataset):
    def __init__(self, input_ids, labels, gender_labels, pad_id=0):
        self.input_ids = input_ids
        self.labels = labels
        self.gender_labels = gender_labels
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
            "gender_labels": self.gender_labels[i]
        }


def get_eval_dataloader(tokenizer, CFG):
    """Loads labeled datasets for analysis."""
    print("Loading Eval Data...")
    with open(CFG.TASKLABS_PATH, "rb") as f:
        tasklabs = pickle.load(f)
    
    genderlabs = torch.load(CFG.GENDLABS_PATH, weights_only=False)
    
    with open(CFG.TENS_DS_PATH, "rb") as f:
        tens_ds = torch.load(f, weights_only=False)

    ad_inputs = tens_ds.tensors[0]
    ad_labels = torch.tensor(tasklabs, dtype=torch.long)
    
    if isinstance(genderlabs, list):
        ad_gender_labels = torch.tensor(genderlabs, dtype=torch.long)
    else:
        ad_gender_labels = genderlabs.to(dtype=torch.long)
    
    train_ds = ActDataset(ad_inputs, ad_labels, ad_gender_labels, pad_id=tokenizer.pad_token_type_id)

    # Simple Random Sampler for Eval (or Weighted if you prefer balanced batches)
    loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, 
        num_workers=4, persistent_workers=True
    )
    return loader

#endregion Data Processing Helper


#region Steering Toolbox

def collect_difference_vectors(model, sae, df, tokenizer, batch_size=32, device="cuda"):
    """ Phase 1: Processes DataFrame to extract feature-wise differences. """
    print("Preparing Data Loader from DataFrame...") #TODO Look into the paired loader
    paired_loader = prepare_paired_dataloader(df, tokenizer, batch_size=batch_size)
    diff_vectors = []
    
    print("STOP")
    print("STOP")

    print("Phase 1: Collecting Difference Vectors...")
    with torch.no_grad():
        for batch in tqdm(paired_loader, desc="Processing Pairs"):
            input_og = batch['original'].to(device)
            input_cf = batch['counterfactual'].to(device)
            
            with model.trace(input_og) as tracer:
                acts_og = model.bert.encoder.layer[6].nns_output[0] 
                sae_latents_og = sae.encode(acts_og)
                sae_latents_og.save()
                
            with model.trace(input_cf) as tracer:
                acts_cf = model.bert.encoder.layer[6].nns_output[0]
                sae_latents_cf = sae.encode(acts_cf)
                sae_latents_cf.save()
            
            vec_og = torch.mean(sae_latents_og.value, dim=1) 
            vec_cf = torch.mean(sae_latents_cf.value, dim=1)
            delta = vec_cf - vec_og
            diff_vectors.append(delta.cpu())

    return torch.vstack(diff_vectors)


def generate_steering_transforms(diff_tensor, mode="mean", n_clusters=3, top_k=10):
    """ Phase 2: Aggregates raw differences into usable steering transforms. """
    print(f"Phase 2: Aggregating using mode '{mode}'...")
    
    if mode == "mean":
        mean_vector = torch.mean(diff_tensor, dim=0)
        var_vector = torch.var(diff_tensor, dim=0)
        top_vars, top_indices = torch.topk(var_vector, k=top_k)
        
        variance_report = {
            "top_k_indices": top_indices.tolist(),
            "top_k_vars": top_vars.tolist()
        }
        return mean_vector, variance_report

    elif mode == "cluster":
        np_diffs = diff_tensor.numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(np_diffs)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        return [centroids[i] for i in range(n_clusters)]

    else:
        raise ValueError(f"Unknown mode: {mode}")


def apply_gated_steering(sae_acts, steering_vector, male_idx, fem_idx, threshold=1.0):
    """
    Applies steering IN-PLACE within the nnsight trace.
    """
    # 1. Gender Salience (Absolute diff between male and female activation)
    male_val = sae_acts[:, :, male_idx]
    fem_val = sae_acts[:, :, fem_idx]
    
    # We use unsqueeze to ensure broadcasting works for the add operation
    gender_salience = torch.abs(male_val - fem_val).unsqueeze(-1)
    
    # 2. Gate Mask
    gate_mask = (gender_salience > threshold).float()
    
    # 3. In-Place Steering
    # NNsight proxies support in-place addition (+=) to modify the graph
    sae_acts += (steering_vector * gate_mask)
    
    return sae_acts

#endregion Steering Toolbox


#region Analysis Suite

def real_analysis(fin_mean, heads, inp_batch):
    """
    Runs regression/classification analysis using trained heads.
    """
    classhead, genderhead = heads
    if classhead is None or genderhead is None:
        print("Skipping Real Analysis (Heads not loaded).")
        return

    # 1. Get Predictions from Heads
    classed_out = classhead(fin_mean)
    gended_out = genderhead(fin_mean)
    
    # 2. Get Ground Truths
    gt_task = inp_batch["labels"].cpu()
    gt_gender = inp_batch["gender_labels"].cpu()

    # 3. Process Preds (Logits -> Binary)
    pred_task = (classed_out["logits"] > 0).long().cpu().flatten()
    pred_gender = (gended_out["logits"] > 0).long().cpu().flatten()

    # 4. Report
    task_acc = (pred_task == gt_task).float().mean().item()
    gender_acc = (pred_gender == gt_gender).float().mean().item()
    
    print(f"\n[Real Analysis] Batch Acc -> Task: {task_acc:.2%} | Gender: {gender_acc:.2%}")
    
    # Optional: Print first few for sanity check
    # for i in range(min(3, len(gt_task))):
    #     print(f"  Ex {i}: GT_G={gt_gender[i]} Pred_G={pred_gender[i]}")


def tn_analysis(pre_mean, model, tokenizer):
    """
    Runs Argmax decoding analysis on the final layer embeddings.
    """
    # We need the value from the proxy
    # pre_mean shape: [Batch, Hidden] or [Batch, Seq, Hidden]
    # If pre_mean is [Batch, Seq, Hidden], take [:, 0, :] for CLS or similar
    
    # Assuming pre_mean is the CLS token or pooled output passed in
    with torch.no_grad():
        # Using the model's lm_head (usually .cls for BERT in HF, 
        # but nnsight might expose it differently depending on architecture)
        # Standard BERT: model.bert is encoder, model.cls is head
        pred_scores = model.cls(pre_mean)
    
    pred_ids = torch.argmax(pred_scores, dim=-1)
    
    # Decode first example in batch
    print(f"[TN Analysis] Decode (Ex 0): {tokenizer.decode(pred_ids[0])}")

#endregion Analysis Suite



def main():
    CFG = Config()
    
    # 1. Load Data for Vector Generation
    print("Loading Gender-Bend Data...")
    genderbent_df = load_og_and_coef_data(CFG.GENDER_BEND_DATA)
    
    print("STOP")
    print("STOP")

    # 2. Load Models
    print(f"Loading NNsight Model ({CFG.MODEL_NAME})...")
    model = LanguageModel(CFG.MODEL_NAME, device_map=CFG.DEVICE, dispatch=True)
    
    print(f"Loading SAE from {CFG.SAE_PATH}...")
    sae = BatchTopKSAE.from_pretrained(CFG.SAE_PATH, device=CFG.DEVICE) 
    model.ae6 = sae # Register for ease of access

    # Load Analysis Heads
    print("Loading Analysis Heads...")
    classhead = load_classhead(CFG.CLASSHEAD_PATH, CFG.DEVICE)
    genderhead = load_classhead(CFG.GENDERHEAD_PATH, CFG.DEVICE)
    analysis_heads = (classhead, genderhead)

    # Tokenizer Setup
    tok = model.tokenizer
    tok.add_special_tokens({'pad_token': '[PAD]'})
    tok.backend_tokenizer.enable_truncation(max_length=CFG.SEQ_LEN)
    tok.backend_tokenizer.enable_padding(length=CFG.SEQ_LEN, pad_id=tok.pad_token_id, pad_token=tok.pad_token)


    # 3. Phase 1: Collect Difference Vectors
    steering_deltas = collect_difference_vectors(
        model=model, 
        sae=sae, 
        df=genderbent_df, 
        tokenizer=tok,
        batch_size=CFG.BATCH_SIZE,
        device=CFG.DEVICE
    )
    
    print(f"Collection Complete. Delta Shape: {steering_deltas.shape}")
    

    print("STOP")
    print("STOP")

    # 4. Phase 2: Generate Steering Vector
    steer_vec, variance_report = generate_steering_transforms(
        steering_deltas, mode="mean", top_k=10
    )
    
    # Ensure vector is on device for steering
    steer_vec = steer_vec.to(CFG.DEVICE)
    
    print(f"Steering Vector Generated. Top Variance Indices: {variance_report['top_k_indices']}")
    
    print("STOP")
    print("STOP")


    # 5. Phase 3: Evaluation Phase (Trace + In-Place Steering + Analysis)
    print("\n--- Starting Evaluation Phase ---")
    
    # Load Evaluation DataLoader (with Labels)
    eval_loader = get_eval_dataloader(tok, CFG)
    
    # Define Submodules for Tracing
    submodule_sae_input = model.bert.encoder.layer[CFG.LAYER_IDX] # Layer 6
    submodule_patch_input = model.bert.encoder.layer[CFG.LAYER_IDX + 1] # Layer 7
    submodule_last = model.bert.encoder.layer[11] # Last Layer

    # Run Eval Loop
    # Just running one batch for testing as requested ("test out the workflow")
    iterator = iter(eval_loader)
    eval_batch = next(iterator)
    
    # Prepare Inputs
    trace_inputs = {
        "input_ids": eval_batch['input_ids'].to(CFG.DEVICE),
        "attention_mask": eval_batch['attention_mask'].to(CFG.DEVICE),
        "token_type_ids": eval_batch['token_type_ids'].to(CFG.DEVICE)
    }

    print("Running Eval Trace...")
    with torch.inference_mode():
        with model.trace(trace_inputs) as tracer:
            
            # A. Encode
            acts = submodule_sae_input.output[0]
            sae_latents = model.ae6.encode(acts)
            
            # B. Apply Steering (IN-PLACE)
            # This modifies 'sae_latents' within the graph
            apply_gated_steering(
                sae_latents, 
                steering_vector=steer_vec, 
                male_idx=CFG.MFEAT, 
                fem_idx=CFG.FFEAT, 
                threshold=1.0 # Tune this
            )
            
            # C. Decode & Patch
            recons = model.ae6.decode(sae_latents)
            submodule_patch_input.input = recons
            
            # D. Save Outputs for Analysis
            # Last layer output (tuple, usually [0] is hidden states)
            l11_out = submodule_last.output[0].save() 
            
    
    # 6. Run Analysis on Saved Outputs
    print("Trace Complete. Running Analysis...")
    
    # Mean pooling for regressors (Match your training logic)
    # l11_out.value shape: [Batch, Seq, Hidden]
    pre_mean = l11_out.value 
    fin_mean = torch.mean(pre_mean, dim=1) 

    # A. Real Analysis (Regressors)
    real_analysis(fin_mean, analysis_heads, eval_batch)

    # B. TN Analysis (Argmax)
    # Pass the CLS token (index 0) to the head for decoding logic
    cls_token_acts = pre_mean[:, 0, :] 
    tn_analysis(cls_token_acts, model, tok)

    print("STOP - Workflow Test Complete")


if __name__ == "__main__":
    main()
