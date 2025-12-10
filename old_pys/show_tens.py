import torch
import plotly.graph_objects as go
import pickle
import numpy as np


#region Funcs



# Recreates full tensor from storage optimised
def recreate_feat_dist(sf_inp: tuple):
	idxs,vals = sf_inp
	ret = torch.zeros([6144])
	ret[idxs] = vals
	return ret


#Takes a recreated feature tensor, and idx of feat of interest, and shows distribution
def show_hist(tens: torch.Tensor, idx: int):
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
def show_active_hist(tens: torch.Tensor, idx: int, show=False, ret=True):
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



#endregion Funcs



#region Main


p = "/home/strah/Documents/pys/blober/outs/feature_1816.pkl"

with open(p,"rb") as f:
    feat = pickle.load(f)

ex = feat["examples"][0][50]

hist_inp = recreate_feat_dist(ex["sparse_features"])
f_idx = 1816


# show_active_hist(hist_inp,f_idx)

print("STOP")
print("STOP")

#endregion Main






