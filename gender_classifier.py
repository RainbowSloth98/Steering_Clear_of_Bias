import torch
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





#region Params

#region? frag_choice

#region? Choices
#? 1 = First chunk only
#? 2 = Second chunk only
#? 3 = Third chunk only; Where only 2 exist, take the latter.
#? 4 = prefinal/penultimate chunk only
#? 5 = final/ultimate chunk only
#endregion
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



#region? act_select

act_select = "middle"
# act_select = "end"

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


device = "cpu"


#endregion Params



#region Paths

root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/gender_classifier/"
load_p = root_p + f"loads/{model_type}/"
save_p = root_p + f"saves/{model_type}/"


#Activations in the shape of [22299,768]
cls_acts_p = load_p+ f"{tok_map[tok_type]}/" +f"v_{frag_choice}/"+ f"l{act_type}_gen_acts_v{frag_choice}.pt"

#Labels of the shape [22299]
labs_p = load_p + "real_malefem_inlp_labs_7_11_25.pt"


#endregion Paths



#region Dataset class definition.

class ActDataset(Dataset):

    def __init__(self, actensors, label):
        # super().__init__()
        self.actensors = actensors
        self.label = label
    

    def __len__(self):
        return len(self.label)


    def __getitem__(self, i):
        return {"input_ids":self.actensors[i],"labels":self.label[i]}



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


def data_balancer(malefem_labs,max_underclass=8401): #8041 female cases (0s)

    mf_zeros = (malefem_labs == 0).nonzero(as_tuple=True)[0]

    #? Create random permutation of idxs for negatives, and only take the amount to equal positives
    mf_zeros = mf_zeros[torch.randperm(len(mf_zeros))][:max_underclass]
    
    #? Same for the positives
    mf_ones = (malefem_labs == 1).nonzero(as_tuple=True)
    mf_ones_r = mf_ones[0][torch.randperm(len(mf_ones[0]))][:max_underclass]

    #Combine the two subgroups
    mf_idxs = torch.cat([mf_zeros,mf_ones_r])

    #Shuffle them
    mf_idxs_r = mf_idxs[torch.randperm(len(mf_idxs))]



    mf_labs_t = malefem_labs[mf_idxs_r] #TODO Need to do acts with mf indexes too for them to line up

    
    return mf_labs_t,mf_idxs_r



#endregion Funcs



#region Loading data and labels

with open(cls_acts_p,"rb") as f:
    full_acts = torch.load(f)
    full_acts = full_acts.to(device)
    # cls_acts = cls_acts.view([23121,-1]).to("cpu") #? Change the view to match the labels


with open(labs_p,"rb") as f:
    full_labs = torch.load(f)
    full_labs = full_labs.to(device)

print("STOP")
print("STOP")

#? Calling the data balancer to downsample the majority class (22299 to about 16k)
labs,idxs = data_balancer(full_labs)
cls_acts = full_acts[idxs]


#endregion Loading data and labels



#region Param 1: Classifier

classifier = WrapClassModel(CustomModCon(hidden_size=h_dim))

#endregion Param 1: Classifier



#region Param 2: Training args

training_args = TrainingArguments(
        output_dir= root_p + f'saves/{model_type}/{tok_map[tok_type]}/l{act_type}_model_v{frag_choice}/',
        eval_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        logging_dir= root_p + f'logs/{model_type}/{tok_map[tok_type]}/l{act_type}_log_v{frag_choice}/',
        report_to="tensorboard",
        logging_steps=10,
        learning_rate=1e-3,
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






























