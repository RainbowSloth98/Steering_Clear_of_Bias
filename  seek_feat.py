import torch
import pickle
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from nnsight import LanguageModel
import random as rand




#region Paths

froot_p = "/home/strah/Documents/Work_related/pys/blober/features/"

mcf_p = "/home/strah/Documents/Work_related/thon-of-py/blober/auto_res/"

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
        print(f"Token ID: {ex['token_id']} ({tok.decode(ex['token_id'])})")
        print(f"Feature value: {ex['value']}")
        print("="*50)
        print("="*50)
    
    else:
        print(f"Feature value: {ex['value']}")
        print(f"Context: {ex['context']}")
        print(f"Token ID: {ex['token_id']} ({tok.decode(ex['token_id'])})")
        print(f"Step: {ex['step']}")
        print("="*50)
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

    print("="*50)
    for i in range(len(exs)):
        print("-*" * 25)
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


#region curr Word search save

#region* Strata Load

cands = []

for i in range(3):
    strat_p = mcf_p + f"strat{i}.pkl"

    with open(strat_p,"rb") as f:
        cands.append(pickle.load(f))


#endregion* Strata Load


#region* Analysis defs

#Find the most common features that responded to list of words
c0 = Counter([cands[0][i][0] for i in range(len(cands[0]))])
c1 = Counter([cands[1][i][0] for i in range(len(cands[1]))])
c2 = Counter([cands[2][i][0] for i in range(len(cands[2]))])

#Most common feats for each strata
#? Because this is only the most common 50, will not include all instances of words.
mcs0 = c0.most_common(50) 
mcs1 = c1.most_common(50) 
mcs2 = c2.most_common(50) 

#Extracting just the features
mc_feats0 = [mcs0[i][0] for i in range(len(mcs0))] 
mc_feats1 = [mcs1[i][0] for i in range(len(mcs1))] 
mc_feats2 = [mcs2[i][0] for i in range(len(mcs2))] 

mcf_set0 = set(mc_feats0)
mcf_set1 = set(mc_feats1)
mcf_set2 = set(mc_feats2)

#Dataframes for fast lookup
df0 = pd.DataFrame(cands[0],columns=["feat","val","tok_id","word"]) 
df1 = pd.DataFrame(cands[1],columns=["feat","val","tok_id","word"]) 
df2 = pd.DataFrame(cands[2],columns=["feat","val","tok_id","word"]) 


#Extract unique word entries
ws0 = [df0[df0["feat"] == i]["word"].unique().tolist() for i in mc_feats0] 
ws1 = [df1[df1["feat"] == i]["word"].unique().tolist() for i in mc_feats1] 
ws2 = [df2[df2["feat"] == i]["word"].unique().tolist() for i in mc_feats2] 


#Checking the intersections between features
inter_1_0 = mcf_set1.intersection(mcf_set0)
inter_2_1 = mcf_set2.intersection(mcf_set1)
inter_2_0 = mcf_set2.intersection(mcf_set0)
pansec = inter_2_1.intersection(inter_1_0)



#endregion* Analysis defs


#region* Analysis

check_feats = [

    682, #0
    2646,#1
    2703,#2
    4949,#3
    1570,#4
    5987,#5
    3881,#6
    4364,#7
    4220,#8
    1043,#9
    2037,#10
    2554,#11
    5052,#12
    2270,#13
    5793,#14
    5581,#15
    4112,#16
    2480,#17
    4980,#18
    3477,#19
    503,#20
    4249,#21
    2746,#22
    4572,#23
]


feat_idxer = dict((v,k) for k,v in dict(enumerate(check_feats)).items())


feat_ldp = "/home/strah/Documents/Work_related/thon-of-py/blober/"

lded_feats = []

for f in tqdm(check_feats,desc="Loading important feats..."):
    p = feat_ldp + f"features/feature_{f}.pkl"
    
    with open(p,"rb") as f:
        lded_feats.append(pickle.load(f))


info_feats = []
nlded_feats = []
for i in range(len(lded_feats)):
    try:
        info_feats.append(fprint_examples(lded_feats[i],ret=True,minimal=False))
    except:
        nlded_feats.append(i) #16,18,20,21


#region** Helper funcs

def feat_words(feat_n): #? The words found here are FROM THE MATCHES WITH THE GENDERED WORDS. Not all the words.
    print(f"Test feat is: {feat_n}")
    print("High:")
    print(df2[df2["feat"] == feat_n]["word"].unique())
    print("Mids")
    print(df1[df1["feat"] == feat_n]["word"].unique())
    print("Lows")
    print(df0[df0["feat"] == feat_n]["word"].unique())

#endregion** Helper funcs

def mc_all_ws(feat,strata):
    ids = []
    for ex in lded_feats[feat_idxer[feat]]["examples"][strata]:
        ids.append(ex["token_id"])
    
    idf = pd.DataFrame(ids)

    mc = Counter(ids)
    unqids = idf[0].unique().tolist()

    return [tok.decode(id) for id in unqids],mc


#curr Workflow: Check dfs for gender info. Check fprint for alignment with dfs.

#k = fprint_examples(feat,ret=True,minimal=True)

# df2[df2["feat"] == 682]["word"].unique()
# df1[df1["feat"] == 682]["word"].unique()
# df0[df0["feat"] == 682]["word"].unique()
#fprint_examples(lded_feats[feat_idxer[test_feat]],minimal=False)





print("STOP")
print("STOP")

#endregion* Analysis


print("STOP")
print("STOP")

#endregion curr Word search save









#region Unused


#region*# Old most common feats


# feats_idxs = [
#     1858, # No immediately understandable feature. Activates highly, so probs important
#     1562, # H - wife,her,daughter,marriage,le; M - Marriage, ##enne, lanka, child, family; L - app, him, bc, app
#     575, # H - ” mostly, decision?; M - Refugee, applicant, of + noise; L - ”,pursuant
#     2273, #! Errored out need to manually look
#     4680, #H - Citizen,(,the,la,can,); M- ],refugee,decision,th?,minister; L- november,file, er, ra
#     904, #! Errored out.
#     5715, # H- xx; M- term, id, for, ##xx; L- the, decisio...n, ',', in
#     5258, #H- ”, ##r/##x; M- translation, 3, audience, sons, L- no,2,dependent,##xx
#     3306, #H- and, or;M- by,),of,(,',';L-file,of,examined,refugee
#     5879,#H- tag, k, da, ant; M- app,refused,fa,reasons,january,after; L- passport, de, division, private, both
#     6002,#H- person, statement,decision,child,decision,refugee,member; M- relationship,act,application,allegations; L- appeal,vo,audience,(,to,tb
#     445,#H-proceeding;M-##dm,humanitarian,',',hearing,##e,to,sponsorship;L-in,(,and,sai,e,the,de
#     1108,#H-saint,freedom,women,god,safety*3;M-convention,returned,gee,th,app,slater,;L-##x,er,##o,decision,for,/
#     1302,#H-“,##rga,##d,conference,complaint,fact;M-citizen,scheduling,k,mail,old,application,stating;L-father,##ellant,##pp,allow,represent,minister,tal
#     3881,#H-person,##ant,##ellant,person,officer;M-##ellant,##nt,counsel,s,is,’;L-her,is,(,solicitor,',',2,app
#     3150,#H-##ster,##er,##ers,dickens;M-##ough,##nt,slater,act,[,minister,disease;L-s,##xx*3..
#     705,#! Errored
#     2669,#H-united,us,hindu,states,you,canadian,ta;M-##oni,la,.,ile,r,canada;L-de,##ellant,##ant,22,77,##xx
#     1242,#H-new,born,became,arrived,birth;M-##bu,permanent,confirmed,n,no,##tal;L-),your,outside,##xx,and
#     4803,#H-minister,barrister,friend,##ellant,consultants,counsel;M-##ent,client,member,app,division,anti,comission;L-##r,##d,heard,app,l,(,address
#     2952,#! Err
#     6124,#H-00,##x,*3,converted;M-06,xx,2011,introduction;L-canada,share,allegedly,...
#     4668,#H-out,with,','*4,and;M-',',form*2,that,',the;L-[,to,##xx,of,s,.
#     3603,#H-s*2,designated*2 s;M- date,##sal,referred,##l,date;L-1 2,[sep],de,purposes
#     359, #H- ',';M- ],.,no,(,s,file,old;L-##ellant,##s,ir,canada,),
#     2819,#H-’,of,”,s;M-For,whom,are,to,(,set;L-age,in,officer,(
#     3121,#H-allegations,whose,adopted,allegations,and;M-.,2021,(,de,(,##ug,
#     997,#H-',',and,.,and; M-conviction,##ant,was,l,##l,##x;L-to,of,audience,e,7th,##xx,and
#     3348,#H-app,off,suffered,stepgather,basis,of,##ig;M-decision,rue,claim,),does,tr,xx;L-t he,##ellant,application,des,e,de,##xx
#     3709,#H-##xx;M-app,app,2007,e,di,application,appeal;L-tribunal,protection,refusal,##re,family
# ]


# feats = []

# for idx in tqdm(feats_idxs,desc="Loading feature files"):
#     fp = froot_p + f"feature_{idx}.pkl"

#     with open(fp,"rb") as f:
#         feats.append(pickle.load(f))


## k = fprint_examples(feat,ret=True,minimal=True)


#endregion*# Old most common feats


#endregion Unused

#endregion Main







