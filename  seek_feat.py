import torch
import pickle
import numpy as np
import plotly.graph_objects as go
from nnsight import LanguageModel
import random as rand




#region Paths

froot_p = "/home/strah/Documents/pys/blober/features/"

#endregion Paths



#region Defs

device = "cuda"
model_name = "bert-base-uncased" #! UNcased, UNCASED
seq_len = 512

#endregion Defs



#region Model

model = LanguageModel(model_name,device_map=device)
# model.config.pad_token_id = tok.pad_token_type_id
model.eval()

#endregion Model



#region Tokenizer
tok = model.tokenizer
tok.add_special_tokens({'pad_token': '[PAD]'})
tok.backend_tokenizer.enable_truncation(max_length=seq_len)
tok.backend_tokenizer.enable_padding(length=seq_len, pad_id=tok.pad_token_id,pad_token=tok.pad_token)
#endregion Tokenizer



#region Funcs


# Recreates full tensor from storage optimised
def recreate_feat_dist(sf_inp: tuple):
	idxs,vals = sf_inp
	ret = torch.zeros([6144])
	ret[idxs] = vals
	return ret


#Takes a recreated feature tensor, and idx of feat of interest, and shows distribution
def hist(tens: torch.Tensor, idx: int):
    """
    Creates a Plotly bar chart of SAE activations.
    
    Args:
        tens (torch.Tensor): Activation tensor of size [6144].
        idx (int): The index of the main feature to highlight.
    """
    # Ensure tensor is on CPU and numpy-compatible
    data = tens.detach().cpu().numpy()
    
    # 1. Define Base Colors (Landscape)
    # specific hex for light grey ensures it fades into background
    colors = np.array(['#E2E2E2'] * len(data)) 
    
    # 2. Identify Top 5 Co-activations (excluding the target idx)
    # We set the target idx to -infinity temporarily so it doesn't get picked as a co-activation
    masked_data = data.copy()
    masked_data[idx] = -np.inf
    
    # Get indices of the top 5 highest values from the rest
    top_5_indices = np.argsort(masked_data)[-5:]
    
    # Highlight Co-activations (e.g., in Blue)
    colors[top_5_indices] = '#636EFA' 
    
    # 3. Highlight the Selected Target Feature (e.g., in Red)
    colors[idx] = '#EF553B'

    # Create the Bar Chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=np.arange(len(data)),
        y=data,
        marker_color=colors,
        # Remove borders on bars for cleaner look at high density
        marker_line_width=0, 
        hoverinfo='x+y',
        name='Activations'
    ))

    # Layout for the "Dashboard" feel
    fig.update_layout(
        title=f"SAE Feature Activations (Target: {idx})",
        xaxis_title="Feature Index",
        yaxis_title="Activation Value",
        template="plotly_white",
        bargap=0, # Removes gaps to show the 'landscape' density better
        showlegend=False
    )

    fig.show()


#Like show_hist, but only shows the non-zero entries.
def active_hist(tens: torch.Tensor, idx: int, show=False, ret=True):
    """
    Displays only non-zero activations, compressed side-by-side.
    Preserves original indices in hover data.
    """
    # 1. Filter Data: Keep only non-zero entries
    # We maintain the tensor on CPU/Numpy for easy indexing
    full_data = tens.detach().cpu().numpy()
    
    # Get indices where activation > 0
    # nonzero returns a tuple of arrays, we want the first one
    nz_indices = np.nonzero(full_data)[0] 
    nz_values = full_data[nz_indices]
    
    # If for some reason the target idx isn't in the non-zero list 
    # (e.g. looking at a dead neuron), we force include it or handle gracefully.
    # Here we assume idx is always active as per your "top entry" description.
    
    # 2. Setup Colors
    # Default color for all active features
    colors = np.array(['#B4B4B4'] * len(nz_indices)) # Slightly darker grey for visibility
    
    # Find where the target 'idx' is located in our new compressed 'nz_indices' array
    # np.where returns a tuple, we take the first element (array of indices) then the first item
    target_loc_array = np.where(nz_indices == idx)[0]
    
    if len(target_loc_array) > 0:
        target_pos = target_loc_array[0]
        colors[target_pos] = '#EF553B' # Red for target
        
        # 3. Find Top 5 Co-activations
        # Create a masked copy of values to find top 5 (excluding the target)
        masked_values = nz_values.copy()
        masked_values[target_pos] = -np.inf
        
        # Get indices of top 5 values in the *compressed* array
        if len(masked_values) > 1: # Only look for co-activations if there are other features
            # argsort sorts ascending, so we take the last 5
            top_5_local_indices = np.argsort(masked_values)[-5:]
            colors[top_5_local_indices] = '#636EFA' # Blue for co-activations
    
    # 4. Create Plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        # X is just 0, 1, 2... N (compressed sequence)
        x=np.arange(len(nz_indices)), 
        y=nz_values,
        marker_color=colors,
        marker_line_width=0,
        
        # 5. Inject the ORIGINAL Feature Indices into the hover data
        customdata=nz_indices,
        hovertemplate=(
            "<b>Feature Index: %{customdata}</b><br>" +
            "Activation: %{y:.4f}<br>" +
            "<extra></extra>" # Removes the secondary box named 'trace 0'
        )
    ))

    fig.update_layout(
        title=f"Active Features (Target: {idx}) - Showing {len(nz_indices)}/{len(full_data)}",
        xaxis_title="Active Features (Ordered by Index)",
        yaxis_title="Activation Value",
        template="plotly_white",
        bargap=0.1, # Slight gap to distinguish distinct bars
        showlegend=False
    )


    if ((not show) and (not ret)):
        raise Exception("At least one of show or ret need to be true")

    if (show):
        fig.show()

    if ret:
        return fig


#Prints one example; Used in fprint_examples as a helper
def _fprint_example(ex,minimal=False):

    if(minimal):
        print("-"*50)
        print(f"Token ID: {ex['token_id']} ({tok.decode(ex['token_id'])})")
        print("-"*50)
        print(f"Feature value: {ex['value']}")
        print("="*50)
    
    else:
        print(f"Feature value: {ex['value']}")
        print("-"*50)
        print(f"Context: {ex['context']}")
        print("-"*50)
        print(f"Token ID: {ex['token_id']} ({tok.decode(ex['token_id'])})")
        print("-"*50)
        print(f"Step: {ex['step']}")
        print("-"*50)
        print("="*50)


#Combines random samples from all of the stratas and print them neatly
def fprint_examples(feat, ret=False,minimal=False):
    #? hard-coded to use 7 randomly sampled examples per strata

    rand.seed(42) #Set seed for reproducibility purposes

    #Getting the indices
    indi_sample_lows = rand.sample(range(len(feat["examples"][0])),7)
    indi_sample_mids = rand.sample(range(len(feat["examples"][1])),7)
    indi_sample_highs = rand.sample(range(len(feat["examples"][2])),7)


    #Creating tuples of (example,index_within_original_strata)
    lows = list(zip([feat["examples"][0][i] for i in indi_sample_lows],indi_sample_lows))
    mids = list(zip([feat["examples"][1][i] for i in indi_sample_mids],indi_sample_mids))
    highs = list(zip([feat["examples"][2][i] for i in indi_sample_highs],indi_sample_highs))


    #Sort them, keeping the indicies
    s_lows = sorted(lows,key= lambda x: x[0]["value"],reverse=True)
    s_mids = sorted(mids,key= lambda x: x[0]["value"],reverse=True)
    s_highs = sorted(highs,key= lambda x: x[0]["value"],reverse=True)

    ss = [s_highs,s_mids,s_lows]
    


    indies = []
    exs = []

    for s in ss: #Treat stratas differently

        temp_indi = []
        temp_exs = []

        for ex,ind in s:
            temp_indi.append(ind)
            temp_exs.append(ex)

        indies.append(temp_indi)
        exs.append(temp_exs)


    strata = [2,1,0]

    for i in range(len(exs)):
        for ex,ind in zip(exs[i],indies[i]):
            print(f"Selected example: {strata[i]}_{ind}")
            _fprint_example(ex,minimal=minimal)


    if(ret == True):
        return exs,indies

def get_hist(feat:int, ex):
    resto_dist = recreate_feat_dist(ex["sparse_features"])

    fig = active_hist(resto_dist,feat,ret=True)

    return fig

#endregion Funcs




#region Main

feat_num = 1816

#Load
feat_p = froot_p + f"feature_{feat_num}.pkl"

with open(feat_p,"rb") as f:
    feat = pickle.load(f)

exs = feat["examples"]

# k = fprint_examples(feat,ret=True,minimal=True)



print("STOP")
print("STOP")


#endregion Main







