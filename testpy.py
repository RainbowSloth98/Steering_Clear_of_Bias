import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

# --- 1. Redefine Custom Classes ---

class CustomModCon(PretrainedConfig):
    def __init__(self, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

class ClassHead(nn.Module):
    def __init__(self, hidden_d):
        super().__init__()
        self.classifier = nn.Linear(hidden_d, 1)

    def forward(self, input_ids=None, labels=None):
        logits = self.classifier(input_ids)
        loss = None

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.view(-1), labels.float())

        return {"logits": logits, "loss": loss}

class WrapClassModel(PreTrainedModel):
    config_class = CustomModCon # explicit link to config

    def __init__(self, config):
        super().__init__(config)
        self.classifier = ClassHead(config.hidden_size)

    def forward(self, input_ids=None, labels=None):
        return self.classifier(input_ids=input_ids, labels=labels)

# --- 2. Load the Model ---

checkpoint_path = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/pytest/loads/l11_checkpoint-1263/"

# Register the custom config so from_pretrained recognizes it if needed
# (Optional but recommended for custom architectures)
CustomModCon.register_for_auto_class()

# Load the model
model = WrapClassModel.from_pretrained(checkpoint_path)
model.eval()



print(f"Model loaded successfully from: {checkpoint_path}")
print(f"Hidden size: {model.config.hidden_size}")



print("STOP")
print("STOP")