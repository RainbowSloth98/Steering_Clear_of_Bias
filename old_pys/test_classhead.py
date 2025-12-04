import torch
import json
import pickle
from transformers import PreTrainedModel,PretrainedConfig
from torch.utils.data import Dataset
from safetensors.torch import load_file
import torch.nn as nn
from nnsight import LanguageModel

#region Params

model_name = "bert-base-uncased"
device = "cuda"
batch_size = 46
data_size = 244214

#endregion




#region Paths

classhead_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/classifier/saves/bert/POOL/model_v2/checkpoint-1734/"
# tens_ds_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/steerer/mf_tens_ds_obj.pt"
cls_acts_p ='/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/classifier/loads/bert/POOL/v_2/all_cls_acts_v2.pt'
# lookup_p = '/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/steerer/mf_ct_df.pkl'


#endregion



#region Classes



class ActDataset(Dataset):

    def __init__(self, actensors, label):
        # super().__init__()
        self.actensors = actensors
        self.label = label
    

    def __len__(self):
        return len(self.label)


    def __getitem__(self, i):
        return {"input":self.actensors[i],"label":self.label[i]}


class ClassHead(nn.Module):
    def __init__(self, hidden_d):
        super().__init__()
        self.classifier = nn.Linear(hidden_d,1)


    def forward(self,input=None,labels=None):
        logits = self.classifier(input)
        loss = None

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            #Labels have to be float, since its expected by BCEWithLogitsLoss
            loss = loss_fn(logits.view(-1), labels.float())


        return {"logits":logits, "loss":loss}


class CustomModCon(PretrainedConfig):
    def __init__(self, hidden_size=768,**kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class WrapClassModel(PreTrainedModel):

    def __init__(self,config):
        super().__init__(config)
        self.classifier = ClassHead(config.hidden_size)

    def forward(self,input=None,labels= None):

        return self.classifier(input=input,labels=labels)



#endregion


#region Load Dataset

#! Not useful anymore
#Keeping a Pandas version of the dataset for lookup and the like
with open(lookup_p,"rb") as f:
	lookup = pickle.load(f)

#?Loading a TensorDataset version of our data
with open(tens_ds_p,"rb") as f:
	tens_ds = torch.load(f,weights_only=False)


with open(cls_acts_p,"rb") as f:
    cls_acts = torch.load(f)
    cls_acts = cls_acts.view([23121,-1]).to("cpu") #? Change the view to match the labels


print("STOP")
print("STOP")


loader = torch.utils.data.DataLoader(
	tens_ds,
	batch_size=batch_size,
	num_workers=4, #Start at this value, later we can move higher up - generally up to number of cores.
	prefetch_factor=8, #Decent to start here - how many batches are loaded at once
	persistent_workers=True,

)


diter = iter(loader)

#endregion







#region Load Model

model = LanguageModel(model_name, device_map=device)
model.eval()

submodule_ref6 = eval("model.bert.encoder.layer[6]") #! The eval allows NNSight to work.


#endregion



#region Load classhead


with open(f"{classhead_p}/config.json") as f:
    cfg_data = json.load(f)

# Create the model directly
model_cfg = CustomModCon(hidden_size=cfg_data["hidden_size"])
classhead = WrapClassModel(model_cfg)

# Load the weights directly
state_dict = load_file(f"{classhead_p}/model.safetensors", device="cpu")
classhead.load_state_dict(state_dict)

classhead.to(device)
classhead.eval()

#endregion



#region Main Code




print("STOP")
print("STOP")

#endregion



print("END FILE")