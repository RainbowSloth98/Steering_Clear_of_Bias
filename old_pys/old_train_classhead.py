#New part
###############################################################
class ClassHead(nn.Module):
    def __init__(self, hidden_d):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_d, hidden_d // 2),  # 768 -> 384
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_d // 2, 1)         # 384 -> 1
        )

    def forward(self, input_ids=None, labels=None):
        logits = self.classifier(input_ids)
        loss = None

        if labels is not None:
            # You can experiment with pos_weight here
            pos_weight = torch.tensor([1.0], device=logits.device) 
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(logits.view(-1), labels.float())

        return {"logits": logits, "loss": loss}


###############################################################
#Old below here
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
#endregion



#region Params

#region? Choices
#? 1 = First chunk only
#? 2 = Second chunk only
#? 3 = Third chunk only; Where only 2 exist, take the latter.
#? 4 = prefinal/penultimate chunk only
#? 5 = final/ultimate chunk only
#endregion
frag_choice = 5




#endregion


#region Paths

root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/classifier/"

load_p = root_p + "loads/"
save_p = root_p + "saves/"


#region* All versions except 0

cls_acts_p = load_p +f"v_{frag_choice}/"+ f"all_cls_acts_v{frag_choice}.pt"
labs_p = load_p +f"v_{frag_choice}/" + "train_labels_23121.pt"

#endregion

#endregion



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


class ClassHead(nn.Module):
    def __init__(self, hidden_d):
        super().__init__()
        self.classifier = nn.Linear(hidden_d,1)


    def forward(self,input_ids=None,labels=None):
        logits = self.classifier(input_ids)
        loss = None

        if labels is not None:
            
            pos_weight = torch.tensor([0.5],device = logits.device) #curr pos_weight: Try lower values

            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            #Labels have to be float, since its expected by BCEWithLogitsLoss
            loss = loss_fn(logits.view(-1), labels.float()) #TODO potentially 


        return {"logits":logits, "loss":loss}
        # return {loss,logits}


class CustomModCon(PretrainedConfig):
    def __init__(self, hidden_size=768,**kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class WrapClassModel(PreTrainedModel):

    def __init__(self,config):
        super().__init__(config)
        self.classifier = ClassHead(config.hidden_size)

    def forward(self,input_ids=None,labels= None):

        return self.classifier(input_ids=input_ids,labels=labels)

#endregion



#region Funcs

#region* monkey patching torch.load

og_torch_load = torch.load

def ntorch_load(*args, **kwargs):
	kwargs['weights_only'] = False
	return og_torch_load(*args, **kwargs)

torch.load = ntorch_load #?its good practice to restore the function when done

#endregion


#endregion



#region Loading data and labels

with open(cls_acts_p,"rb") as f:
    cls_acts = torch.load(f)
    cls_acts = cls_acts.view([23121,-1]) #? Change the view to match the labels


with open(labs_p,"rb") as f:
    labs = torch.load(f)

#endregion



#region Param 1: Classifier

classifier = WrapClassModel(CustomModCon(hidden_size=768))

#endregion



#region Param 2: Training args

training_args = TrainingArguments(
        output_dir=f'/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/classifier/saves/model_v{frag_choice}/',
        eval_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        logging_dir=f'/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/classifier/logs/log_v{frag_choice}/',
        report_to="tensorboard",
        logging_steps=10,
        learning_rate=1e-3,#curr Try this, reduce a bit as we go on.
)

#endregion



#region Params 3/4, training and eval dataset

class_ds = ActDataset(cls_acts,labs)
num_samples = len(class_ds)
train_size = int(0.8*num_samples)

train_ds, eval_ds = torch.utils.data.random_split(class_ds,[train_size,num_samples-train_size])

#endregion



#region Param 5: Metrics to compute
def compute_metrics(eval_pred):
    
    logits,labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()
    labels = torch.tensor(labels).int()
    accuracy = (preds == labels).float().mean().item()
    
    f1 = f1_score(labels,preds)
    precision = precision_score(labels,preds)
    recall = recall_score(labels,preds)
    bin_acc = accuracy_score(labels,preds)

    return {
        "accuracy":accuracy,
        "f1":f1,
        "precision":precision,
        "recall":recall,
        "bin_acc":bin_acc
        }

#endregion



#region# Tester

# lp = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/full_acts_extract/loads/our_train.pkl"

# with open(lp,"rb") as f:
#     l = pickle.load(f)




# print("STOP")
# print("STOP")


#endregion



#region# Debugging data

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


#endregion




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

#endregion 






