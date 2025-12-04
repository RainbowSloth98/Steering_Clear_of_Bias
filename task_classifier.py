import pandas as pd
import pickle
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedModel,PretrainedConfig

#Helpful for diagnostics
import pyperclip as clip
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


#region? Info
#? Loading the already extracted CLS tokens from the final layer of BERT
#? to train the classification head to use for downstream purposes
#? Additionally, to also serve as a POC that it can classify well.

#TODO OG batches were 157
#endregion? Info



#region Params

#region? frag_choice

#region* Choices
#? 1 = First chunk only
#? 2 = Second chunk only
#? 3 = Third chunk only; Where only 2 exist, take the latter.
#? 4 = prefinal/penultimate chunk only
#? 5 = final/ultimate chunk only
#endregion* Choices
frag_choice = 2

#endregion? frag_choice


#region? tok_type
tok_type = "pool" 
# tok_type = "cls"
tok_map = {"cls":"CLS","pool":"POOL"}
#endregion


#region? model_type

model_type = "bert"
# model_type = "bge" 

if(model_type == "bert"):
    h_dim = 768 #for bert
elif(model_type == "bge"):
    h_dim = 1024 #for bge


#endregion? model_type



#Positive weight, since the dataset is unbalanced. 1.25 works the best in our experience.
# pos_w = 1.25 #? Explicitly balancing the dataset now by undersampling the majority class

#endregion Params



#region Paths

root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/task_classifier/"

load_p = root_p + "loads/"


cls_acts_p = load_p + f"{model_type}/{tok_map[tok_type]}/v_{frag_choice}/all_cls_acts_v{frag_choice}.pt"
labs_p = load_p + "train_labels_23121.pt"

#endregion Paths



#region Dataset class definition.


#region* linear_class_head

class ClassHead(nn.Module):
    def __init__(self, hidden_d):
        super().__init__()
        self.classifier = nn.Linear(hidden_d,1)


    def forward(self,input_ids=None,labels=None):
        logits = self.classifier(input_ids)
        loss = None

        if labels is not None:
            
            # pos_weight = torch.tensor([pos_w],device = logits.device) # pos_weight: Try lower values
            # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            loss_fn = nn.BCEWithLogitsLoss()
            #Labels have to be float, since its expected by BCEWithLogitsLoss
            loss = loss_fn(logits.view(-1), labels.float()) #TODO potentially 


        return {"logits":logits, "loss":loss}
        # return {loss,logits}

#endregion* linear_class_head


#region*# MLP_class_head

# class ClassHead(nn.Module):
    # def __init__(self, hidden_d):
    #     super().__init__()
    #     self.classifier = nn.Sequential(
    #         nn.Linear(hidden_d, hidden_d // 2),  # 768 -> 384
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Linear(hidden_d // 2, 1)         # 384 -> 1
    #     )

    # def forward(self, input_ids=None, labels=None):
    #     logits = self.classifier(input_ids)
    #     loss = None

    #     if labels is not None:
    #         # You can experiment with pos_weight here
    #         pos_weight = torch.tensor([0.5], device=logits.device) #curr Set pos weight
    #         loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #         loss = loss_fn(logits.view(-1), labels.float())

    #     return {"logits": logits, "loss": loss}

#endregion*# MLP_class_head



class ActDataset(Dataset):

    def __init__(self, actensors, label):
        # super().__init__()
        self.actensors = actensors
        self.label = label
    

    def __len__(self):
        return len(self.label)


    def __getitem__(self, i):
        return {"input_ids":self.actensors[i],"labels":self.label[i]}


class CustomModCon(PretrainedConfig):
    def __init__(self, hidden_size=h_dim,**kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class WrapClassModel(PreTrainedModel):

    def __init__(self,config):
        super().__init__(config)
        self.classifier = ClassHead(config.hidden_size)

    def forward(self,input_ids=None,labels= None):

        return self.classifier(input_ids=input_ids,labels=labels)

#endregion Dataset class definition.



#region Funcs

#region* monkey patching torch.load

og_torch_load = torch.load

def ntorch_load(*args, **kwargs):
	kwargs['weights_only'] = False
	return og_torch_load(*args, **kwargs)

torch.load = ntorch_load #?its good practice to restore the function when done

#endregion


def data_balancer(labs,max_underclass=6587): #6587 accept cases (0s)

    #? Random permutation based on the amount in the underclass
    mf_zeros = (labs == 0).nonzero(as_tuple=True)[0]
    mf_zeros = mf_zeros[torch.randperm(len(mf_zeros))][:max_underclass]
    
    #? Same for the positives
    mf_ones = (labs == 1).nonzero(as_tuple=True)
    mf_ones_r = mf_ones[0][torch.randperm(len(mf_ones[0]))][:max_underclass]

    #Combine the two subgroups
    mf_idxs = torch.cat([mf_zeros,mf_ones_r])

    #Shuffle them
    mf_idxs_r = mf_idxs[torch.randperm(len(mf_idxs))]



    mf_labs_t = labs[mf_idxs_r] #TODO Need to do acts with mf indexes too for them to line up

    
    return mf_labs_t,mf_idxs_r


#endregion Funcs



#region Loading data and labels

with open(cls_acts_p,"rb") as f:
    full_acts = torch.load(f)
    full_acts = full_acts.view([23121,-1]).to("cpu") #? Change the view to match the labels


with open(labs_p,"rb") as f:
    full_labs = torch.load(f)



#Undersampling the majority class
labs,idxs = data_balancer(full_labs)
cls_acts = full_acts[idxs]


#endregion Loading data and labels



#region Param 1: Classifier

classifier = WrapClassModel(CustomModCon(hidden_size=h_dim))

#endregion Param 1: Classifier



#region Param 2: Training args

training_args = TrainingArguments(
        output_dir=f'/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/task_classifier/saves/{model_type}/{tok_map[tok_type]}/model_v{frag_choice}/',
        eval_strategy='epoch',
        save_strategy='epoch',
        # num_train_epochs=3,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        logging_dir=f'/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/task_classifier/logs/{model_type}/{tok_map[tok_type]}/log_v{frag_choice}/',
        report_to="tensorboard",
        logging_steps=10,
        learning_rate=1e-2
)

#endregion Param 2: Training args



#region Params 3/4, training and eval dataset

class_ds = ActDataset(cls_acts,labs)
num_samples = len(class_ds)
train_size = int(0.8*num_samples)

train_ds, eval_ds = torch.utils.data.random_split(class_ds,[train_size,num_samples-train_size])

#endregion Params 3/4, training and eval dataset



#region Param 5: Metrics to compute

def compute_metrics(eval_pred):
    
    logits,labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().view(-1)
    labels = torch.tensor(labels).int().view(-1)


    
    
    f1 = f1_score(labels,preds)
    precision = precision_score(labels,preds)
    recall = recall_score(labels,preds)
    bin_acc = accuracy_score(labels,preds)

    return {
        "f1":f1,
        "precision":precision,
        "recall":recall,
        "bin_acc":bin_acc
        }

#endregion Param 5: Metrics to compute



#region# Visualising data with T-SNE

# num_samples = 10000
# indices = np.random.permutation(len(cls_acts))[:num_samples]
# X_sample = cls_acts[indices].cpu().numpy()
# y_sample = labs[indices].cpu().numpy()

# print("Running t-SNE... this may take a few minutes.")
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(X_sample)

# # Plot the results
# plt.figure(figsize=(12, 8))
# scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=y_sample, cmap='coolwarm', alpha=0.6)
# plt.title('t-SNE Visualization of CLS Activations')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.legend(handles=scatter.legend_elements()[0], labels=['Negative Class (0)', 'Positive Class (1)'])
# plt.show()


#endregion# Visualising data with T-SNE




#region Main Training

trainer = Trainer(
    model=classifier,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics
)


trainer.train()



print("STOP")
print("STOP")

#endregion Main Training 





