import torch
import pickle
import pandas as pd
from collections import Counter, defaultdict
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
def active_hist(tens: torch.Tensor, idx: int, topk = 5, show=False, ret=True):
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
			top_local_indices = np.argsort(masked_values)[-topk:]
			colors[top_local_indices] = '#636EFA' # Blue for co-activations
	
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
		return fig,top_local_indices



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

	samp_init = 7

	tosamp_highs = samp_init
	tosamp_mids = samp_init
	tosamp_lows = samp_init
	

	if(len(feat["examples"][2]) == 0):
		s_highs = []
	else:
		if (len(feat["examples"][2]) < tosamp_highs):
			tosamp_highs = len(feat["examples"][2])

		indi_sample_highs = rand.sample(range(len(feat["examples"][2])),tosamp_highs)#Getting the indices
		highs = list(zip([feat["examples"][2][i] for i in indi_sample_highs],indi_sample_highs))#Tupling (example,index_within_original_strata)
		s_highs = sorted(highs,key= lambda x: x[0]["value"],reverse=True) #Sort them, keeping the indicies

	#Do for mids too

	if(len(feat["examples"][1]) == 0):
		s_mids = []
	else:

		if (len(feat["examples"][1]) < tosamp_mids):
			tosamp_mids = len(feat["examples"][1])

		indi_sample_mids = rand.sample(range(len(feat["examples"][1])),tosamp_mids)
		mids = list(zip([feat["examples"][1][i] for i in indi_sample_mids],indi_sample_mids))
		s_mids = sorted(mids,key= lambda x: x[0]["value"],reverse=True)
	
	#Do for highs too

	if(len(feat["examples"][0]) == 0):
		s_lows = []
	else:

		if (len(feat["examples"][0]) < tosamp_lows):
			tosamp_lows = len(feat["examples"][0])

		indi_sample_lows = rand.sample(range(len(feat["examples"][0])),tosamp_lows)
		lows = list(zip([feat["examples"][0][i] for i in indi_sample_lows],indi_sample_lows))
		s_lows = sorted(lows,key= lambda x: x[0]["value"],reverse=True)
	
	

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


#region TODO Word search, Feat Interp, Co-acts


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


print("STOP")
print("STOP")

#endregion* Analysis defs


#region* Analysis



#region**# T2.1: Look at co-acts as well.

# nfeats = [
# 	170,
# 	4364,
# 	1599,
# 	4803,
# 	714,
# 	3348,
# 	632,
# 	461,
# 	948,
# 	6132,
# 	3924,
# 	5782,
# 	2532,
# 	4026,
# 	3881,
# 	2270,
# 	4868,
# 	5196,
# 	926,
# 	1041,
# 	6002,
# ]


# nfeat_idxer = dict((v,k) for k,v in dict(enumerate(nfeats)).items())

# n_lded_feats = []
# for f in tqdm(nfeats,desc="Loading feats..."):
#     p = feat_ldp + f"features/feature_{f}.pkl"
	
#     with open(p,"rb") as f:
#         n_lded_feats.append(pickle.load(f))




# print("STOP")
# print("STOP")

#endregion**# Look at co-acts as well.


#region**# T3.0: Finding FemFeat from FemWords

# fem_words= [
# "female",
# "woman",
# "girl",
# "mrs",
# "ms","miss",
# "madam",
# "lady",
# "wife",
# "mother",
# "daughter",
# "sister",
# "grandmother",
# "granddaughter",
# "widow",
# "matriarch",
# ]


# wls0 = defaultdict(list)
# for c in cands[0]:
# 	wls0[c[-1]].append(c[0])
# wcs0 = {w: Counter(wls0[w]) for w in fem_words}

# wls1 = defaultdict(list)
# for c in cands[1]:
# 	wls1[c[-1]].append(c[0])
# wcs1 = {w: Counter(wls1[w]) for w in fem_words}

# wls2 = defaultdict(list)
# for c in cands[2]:
# 	wls2[c[-1]].append(c[0])
# wcs2 = {w: Counter(wls2[w]) for w in fem_words}


# fem_strata_count = {}
# fem_tot_count = {}

# for w in fem_words:
#     fem_strata_count[w] = [
#         wcs0[w].most_common(50),
#         wcs1[w].most_common(50),
#         wcs2[w].most_common(50),
#         ]
	
#     fem_tot_count[w] = wcs0[w]+wcs1[w]+wcs2[w]

# fem_high_agg = sum((wcs2[w] for w in fem_words), Counter())
# fem_mid_agg = sum((wcs1[w] for w in fem_words), Counter())
# fem_low_agg = sum((wcs0[w] for w in fem_words), Counter())



# 492,781,3796,738,1046,4016,3990,2558,5939,5761,904
# 	"wife" mid counts >50

# (170, 578), (1562, 558), (632, 517), (2270, 304), (714, 137), (1186, 92), (3881, 83), (425, 71)
# 	fem_high_agg > 50

# (492, 574), (993, 499), (1046, 498), (3796, 485), (3569, 369), (3089, 283), (781, 255), (738, 220), (2746, 161), (3990, 112), (4016, 108), (2558, 105), (5939, 97), (4803, 94), (2270, 86), (5716, 86), (632, 79), (904, 79), (165, 77), (3287, 76), (2532, 71), (1421, 63), (4572, 59), (2896, 58), (4980, 57)
# 	fem_mid_agg > 50

# [(682, 299), (1134, 26), (2703, 21), (1142, 19), (4711, 19),
# 	fem_low_agg




#endregion**# Finding FemFeat from FemWords


#region** Loading Features



#region**# T1.1: Generally gendered features

# check_feats = [
#     682, #0 Female Pronoun Feat
#     2646,#1 Pronouns (and quazi-pronouns) as grammatical objects; Compare with 4220, and 2037
#     2703,#2 First Person feature, mostly judge
#     4949,#3 Appositive feat; Clarifyication after "," or in brackets
#     1570,#4 General pronoun detector
#     5987,#5 Narrative struct feat; Supresses legal jargon; Difference between story and law
#     3881,#6 Animacy/Person feature
#     4364,#? 7 The masculine feature, really interesting and important.
#     4220,#8 Narrative object detector feature. Compare to 2646.
#     1043,#9 Tracks person and sphere of influence, identity/agency/ownership
#     2037,#10 Grammatical Subject detector
#     2554,#11 Self-referentail pronouns, e.g. herself, himself etc.
#     5052,#12 The most general purpose anafora (standin word)
#     2270,#13 Relational nouns, the "possessed", e.g. the "brother" in "I am going with my brother"
#     5793,#14 Semantic relation, connection and comparison; (with, to ,after, joint) etc.
#     5581,#15 Asylum seeking detector
#     4112,#16 /////////////////////////////////////
#     2480,#17 the "Because" feture. Logical causation
#     4980,#18 /////////////////////////////////////
#     3477,#19 mutual interaction detector; Mostly in idioms like "one anoter" and "each other"
#     503,#20 Sentence initial He.
#     4249,#21 "Zero Relative Clause, no Pronoun" detector, e.g. "...woman [that] he..."
#     2746,#22 "Quantity and Magnitude" detector. At high deals with quants, at mid deals with things that have a quantitive element to them
#     4572,#23 Identity feature; Is it the same, different or composed?
# ]

#endregion**# T1: Generally gendered features



#region**# T3.1: Female only feats - Looking for female feature.


# check_feats = [
#     1562, #? This is the female feature!!!
#     1186, 
#     425, 
#     492, 
#     993, 
#     1046, 
#     3796, 
#     3569, 
#     3089, 
#     781, 
#     738, 
#     3990, 
#     4016, 
#     2558, 
#     5939, 
#     5716, 
#     904, 
#     165, 
#     3287, 
#     1421, 
#     2896,
# ]



#endregion**# T3.1: Female only feats - Looking for female feature.


#? T4: Male and feamle feature focus
check_feats = [
	4364, #Main male feature
	1562, #Main female feature
]
mfeat = check_feats[0]
ffeat = check_feats[1]



#region**# T4.1: Coact interps

# check_feats = [
# 	5223,
# 	4810,
# 	3499,
# 	6079,
# 	1624,
# 	4680,

# 	4992,
# 	2144,
# 	741,
# 	5549,
# 	3790,
# 	4303,
# 	2322,
# 	5715,
# 	2159,
# 	1302,
# 	5336,
# 	3006,

# 	644,
# 	4516,
# 	2106,
# 	69,
# 	5732,
# 	4083,
# 	185,
# 	1434,
# ]

#endregion**# T4.1: Coact interps


feat_idxer = dict((v,k) for k,v in dict(enumerate(check_feats)).items())
feat_ldp = "/home/strah/Documents/Work_related/thon-of-py/blober/"


#Load the features
lded_feats = []
for f in tqdm(check_feats,desc="Loading important feats..."):
	p = feat_ldp + f"features/feature_{f}.pkl"
	
	with open(p,"rb") as f:
		lded_feats.append(pickle.load(f))



#Check for the ones that loaded properly, and the ones that didn't
info_feats = []
nlded_feats = []
for i in range(len(lded_feats)):
    try:
        info_feats.append(fprint_examples(lded_feats[i],ret=True,minimal=False))
    except:
        nlded_feats.append(i)



# tprint(test_feat,topk=30)
# ctx_word = ""
# ctxs = all_find_ex_ctx(test_feat,ctx_word)
# _ctxs_len(ctxs)
# _ctxs_sample(ctxs,s)
# _find_ex_ctx(test_feat,ctx_word,s)


#region* Helper funcs


#region** T1 Helpers


#? Turns out this one isn't very useful for feat interps generally...
def feat_words(feat_n): # Gender related mathes only.
	print(f"Test feat is: {feat_n}")
	print("High:")
	print(df2[df2["feat"] == feat_n]["word"].unique())
	print("Mids")
	print(df1[df1["feat"] == feat_n]["word"].unique())
	print("Lows")
	print(df0[df0["feat"] == feat_n]["word"].unique())


#region* Context related


def _find_ex_ctx(feat,w,strata): #Index and context of example containing a word
	
	search_id = tok.encode(w)[1] #Encoding starts with 101, ends with 102
	
	hit_ids = []
	hit_ctxs = []

	for i in range(len(lded_feats[feat_idxer[feat]]["examples"][strata])):
		if (lded_feats[feat_idxer[feat]]["examples"][strata][i]["token_id"] == search_id):
			hit_ids.append(i)
			hit_ctxs.append(lded_feats[feat_idxer[feat]]["examples"][strata][i]["context"])

	return hit_ids,hit_ctxs


def _ctxs_len(ctxs):
	for i in range(3):
		print(f"Strata {i}: {len(ctxs[i][0])}")


def _ctxs_sample(ctxs,strata,k=5):
	for i in range(k):
		print(f"Ex {i}")
		print(ctxs[strata][1][i])
		print("-"*20)


def all_find_ex_ctx(feat,w):
	
	ret = []

	for i in range(3):
		ret.append(_find_ex_ctx(feat,w,i))
	
	return ret


#endregion* Context related


#region* Most common related


def mc_all_ws(feat,strata): #All Unique words and Counter for feat

	if (len(lded_feats[feat_idxer[feat]]["examples"][strata]) == 0): #Catch empties
		return []

	ids = []
	for ex in lded_feats[feat_idxer[feat]]["examples"][strata]:
		ids.append(ex["token_id"])
	
	idf = pd.DataFrame(ids)
	unqids = idf[0].unique().tolist()
	
	#Creating the counter
	mc = Counter(ids) #Count ids
	cdict = {(tok.decode(k),v) for k,v in mc.items()} #dict with names instead
	redict = sorted(cdict,key=lambda x: x[1],reverse=True) #sort by most-common-first
	

	return [tok.decode(id) for id in unqids],redict


def mc_awas(feat): #most common, all words all strata
	ret = []

	for i in range(3):
		ret.append(mc_all_ws(feat,i))

	return ret


#Helper function for printing mcawas
def _mc_print(mcawa,topk):
	print("mc_Highs")
	if(mcawa[2] == []):
		print("Empty!")
	else:
		print(mcawa[2][1][:topk])
	print("mc_Mids")
	if(mcawa[1] == []):
		print("Empty!")
	else:
		print(mcawa[1][1][:topk])
	print("mc_Lows")
	if(mcawa[0] == []):
		print("Empty!")
	else:
		print(mcawa[0][1][:topk])



#endregion* Most common related


def tprint(feat,topk=10): #Complete print function for all of the above
	mcawas = mc_awas(feat)
	
	print(f"Feature: {feat}")
	print(f"All words, and most common {topk} by count")
	_mc_print(mcawas,topk)
	print("#"*50)
	print("#"*50)
	print("Example sample, with context")
	fprint_examples(lded_feats[feat_idxer[feat]],minimal=False)



#endregion** T1 Helpers


#region** T2 Helpers


def gen_mc_all_ws(feat,strata,lded_feats,feat_idxer): #All Unique words and Counter for feat

	if (len(lded_feats[feat_idxer[feat]]["examples"][strata]) == 0): #Catch empties
		return []

	ids = []
	for ex in lded_feats[feat_idxer[feat]]["examples"][strata]:
		ids.append(ex["token_id"])
	
	idf = pd.DataFrame(ids)
	unqids = idf[0].unique().tolist()
	
	#Creating the counter
	mc = Counter(ids) #Count ids
	cdict = {(tok.decode(k),v) for k,v in mc.items()} #dict with names instead
	redict = sorted(cdict,key=lambda x: x[1],reverse=True) #sort by most-common-first
	

	return [tok.decode(id) for id in unqids],redict


def gen_mc_awas(feat,lded_feats,feat_idxer): #most common, all words all strata
	ret = []

	for i in range(3):
		ret.append(mc_all_ws(feat,i,lded_feats,feat_idxer))

	return ret


def ntprint(feat,n_lded_feats,nfeat_idxer,topk=10): #Complete print function for all of the above
	mcawas = gen_mc_awas(feat,lded_feats,feat_idxer)
	
	print(f"Feature: {feat}")
	print(f"All words, and most common {topk} by count")
	_mc_print(mcawas,topk)
	print("#"*50)
	print("#"*50)
	print("Example sample, with context")
	fprint_examples(n_lded_feats[nfeat_idxer[feat]],minimal=False)


#endregion** T2 Helpers


#region** T4 helpers


#Return co-acts on specific word between 2 feats
def coact_recons(feat_main,feat_check,w,s):
	
	main_hits,_ = _find_ex_ctx(feat_main,w,s)

	rec_feats = []
	for i in main_hits:
		rec_feats.append(recreate_feat_dist(lded_feats[feat_idxer[feat_main]]["examples"][s][i]["sparse_features"]))

	return [(rec_feats[i][feat_main],rec_feats[i][feat_check]) for i in range(len(rec_feats))]


def avg(l):
    return sum(l)/len(l)

def word_comps (feat1,feat2,word_list,s):

	comps = []
	for w in word_list:
		comps.append(coact_recons(feat1,feat2,w,s))
		# difs = [c[i][0]-c[i][1] for i in range(len(c))]
		# avgs.append(avg(difs))
	
	return comps






#endregion** T4 helpers


#endregion* Helper funcs





print("STOP")
print("STOP")

#endregion** Loading Features



#region curr Finding co-acts for male and female

mfeat,ffeat = check_feats


mload = lded_feats[0]
fload = lded_feats[1]


#region** Word defs

#Male and female words
mw_search = {
	"male",
	"man",
	"boy",
	"sir",
	"son",
	"husband",
	"father",
	"brother",
	"widower",
	"fiancé",
	"mr",
}
mw_toks = {tok.encode(w)[1] for w in mw_search}

fw_search = {
	"female",
	"woman",
	"girl",
	"madam",
	"wife",
	"mother",
	"sister",
	"daughter",
	"widow",
	"fiancée",
	"mrs",
	"ms",
	"miss",
}
fw_toks = {tok.encode(w)[1] for w in fw_search}


#Male and female pronouns - these will likely be a separate category?
mp_search = {
	"he",
	"him",
	"his",
	"himself",
}
mp_toks = {tok.encode(w)[1] for w in mp_search}

fp_search = {
	"she",
	"her",
	"hers",
	"herself",
}
fp_toks = {tok.encode(w)[1] for w in fp_search}

#endregion** Word defs


mw_collect = [[],[],[]]
mp_collect = [[],[],[]]
for i in range(len(mload["examples"])):
	for ex in mload["examples"][i]:
		if (ex["token_id"]) in mw_toks:
			mw_collect[i].append(recreate_feat_dist(ex["sparse_features"]))

		if(ex["token_id"]) in mp_toks:
			mp_collect[i].append(recreate_feat_dist(ex["sparse_features"]))


fw_collect = [[],[],[]]
fp_collect = [[],[],[]]
for i in range(len(fload["examples"])):
	for ex in fload["examples"][i]:
		if (ex["token_id"]) in fw_toks:
			fw_collect[i].append(recreate_feat_dist(ex["sparse_features"]))

		if(ex["token_id"]) in fp_toks:
			fp_collect[i].append(recreate_feat_dist(ex["sparse_features"]))


#stack them, for efficiency
for i in range(3):
	if(len(mw_collect[i]) != 0):
		mw_collect[i] = torch.stack(mw_collect[i])
	if(len(mp_collect[i]) != 0):
		mp_collect[i] = torch.stack(mp_collect[i])
	if(len(fw_collect[i]) != 0):
		fw_collect[i] = torch.stack(fw_collect[i])
	if(len(fp_collect[i]) != 0):
		fp_collect[i] = torch.stack(fp_collect[i])


#region** Helpers T4


def rec_len (l):
    for t in l:
        print(len(t))

def coact_counts(l_t: list[torch.Tensor], target_idx: int):
    """
    Analyzes a list of tensors to count feature co-occurrences stratified by the 
    activation magnitude of a target index.

    Args:
        l_t: A list of PyTorch tensors. Can be 1D (single sample) or 2D (batch of samples).
        target_idx: The index of the feature of interest.

    Returns:
        tuple: (low_counts, mid_counts, high_counts)
    """
    # Initialize counters for each strata
    low_counts = Counter()
    mid_counts = Counter()
    high_counts = Counter()

    # Strata definitions (Lower bound inclusive, Upper bound exclusive)
    # Note: We ignore activations < 0.2 based on the "Low" definition provided.
    LOW_RANGE = (0.2, 1.5)
    MID_RANGE = (1.5, 4.0)
    HIGH_THRESHOLD = 4.0

    # Iterate through the list with tqdm
    for tensor in tqdm(l_t, desc=f"Analyzing co-activations for feature {target_idx}"):
        
        # Ensure we are working with a 2D batch for consistent logic
        # If input is 1D [num_features], make it [1, num_features]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        # 1. Get the activation values of the target feature for this batch
        target_acts = tensor[:, target_idx]

        # 2. Create masks for each strata based on the target feature's value
        # Low: 0.2 <= val < 1.5
        mask_low = (target_acts >= LOW_RANGE[0]) & (target_acts < LOW_RANGE[1])
        
        # Mid: 1.5 <= val < 4.0
        mask_mid = (target_acts >= MID_RANGE[0]) & (target_acts < MID_RANGE[1])
        
        # High: val >= 4.0
        mask_high = (target_acts >= HIGH_THRESHOLD)

        # 3. Helper to update counters for a specific mask
        def update_strata_counter(mask, counter_obj):
            if not mask.any():
                return
            
            # Select only the samples where the target feature falls in this strata
            relevant_samples = tensor[mask]
            
            # Find indices of ALL non-zero elements in these samples
            # nonzero() returns [sample_idx, feature_idx]
            # We only care about feature_idx (column 1)
            co_occurring_indices = torch.nonzero(relevant_samples)[:, 1]
            
            # Optional: Filter out the target_idx itself so it doesn't dominate the counts
            co_occurring_indices = co_occurring_indices[co_occurring_indices != target_idx]
            
            # Update the counter
            counter_obj.update(co_occurring_indices.tolist())

        # Update each counter
        update_strata_counter(mask_low, low_counts)
        update_strata_counter(mask_mid, mid_counts)
        update_strata_counter(mask_high, high_counts)

    return low_counts, mid_counts, high_counts


def mc(counters, topk):
    for i in range(3):
        print(f"Strata {i}")
        print(counters[i].most_common(topk))
        print("#"*25)

#endregion** Helpers T4



mmask = (mw_collect[2] > 4.0 )
mmask[:,mfeat] = False

fmask = (fw_collect[2] > 4.0)
fmask[:,ffeat] = False
#region* Experiment: How often do co-acts occur as high



male_tot = 3280
fem_tot = 551

male_edit = [[170, 2230], [948, 1310], [1599, 1271], [4803, 992], [5223, 726], [714, 668], [4810, 577], [3348, 403], [632, 334], [3499, 332], [4534, 298], [2270, 280], [6079, 219], [4680, 174], [1624, 143], [2708, 100], [3101, 67], [5782, 63], [1520, 43], [4756, 36], [4868, 34], [4934, 29], [3881, 26], [6002, 23], [3398, 20], [2144, 19], [5336, 13], [3709, 12], [3006, 11], [457, 9], [6132, 8], [461, 7], [3790, 7], [5678, 7], [2532, 6], [3924, 6], [1242, 5], [4992, 5], [2159, 4], [5005, 3], [2491, 3], [5715, 3], [5549, 3], [1735, 3], [741, 2], [1041, 2], [2026, 2], [4303, 1], [1302, 1], [2322, 1]]

fem_edit = [[170, 333], [1599, 255], [948, 192], [5223, 133], [3881, 101], [4810, 89], [3348, 80], [2270, 69], [3499, 55], [1624, 54], [632, 52], [6079, 40], [4803, 40], [4680, 29], [6002, 26], [2708, 25], [4868, 14], [5005, 14], [5678, 14], [4756, 14], [3101, 13], [4534, 12], [6132, 9], [714, 7], [2491, 7], [1186, 7], [4083, 6], [425, 5], [4934, 4], [1735, 3], [5782, 3], [1520, 3], [5732, 2], [1242, 2], [5793, 2], [2026, 1], [3709, 1], [3398, 1], [4516, 1], [69, 1], [644, 1], [457, 1], [1434, 1], [185, 1], [2106, 1]]

for i in range(len(male_edit)):
    male_edit[i][1] = male_edit[i][1] / male_tot


for i in range(len(fem_edit)):
    fem_edit[i][1] = fem_edit[i][1] / fem_tot

#endregion* Experiment: How often do co-acts occur as high



print("STOP")
print("STOP")

#endregion curr Finding co-acts for male and female




#endregion* Analysis







#endregion TODO Word search, Feat Interp, Co-acts









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







