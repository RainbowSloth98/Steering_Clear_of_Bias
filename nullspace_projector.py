import torch
import numpy as np
import scipy
from typing import List, Dict
from tqdm import tqdm
import random
import warnings
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression 

from nnsight import LanguageModel
from nnsight.module import Module
import pickle
import gc
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

torch.manual_seed(42)


#region curr Testing

ld_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/INLP/loads/tpickfull.pkl"

with open(ld_p,"rb") as f:
	tpickfull = pickle.load(f)




print("STOP")
print("STOP")

#endregion curr




#region Params


#region? model_type
model_type = "bert"
# model_type = "bge"
#endregion?

#region? tok_type
tok_type = "pool" #? Admittedly, we'll probably only be using pool, since CLS doesn't classify
# tok_type = "cls"
#endregion?

#region? f_c
f_c = 2 #Use only 2nd
# f_c = 3 #Use only 3rd
# f_c = 4 #Use second to last
# f_c = 5 #Use last
#endregion?


#endregion Params



#region Defs


#region* Standard defs

tok_map = {"cls":"CLS","pool":"POOL"}

if(model_type == "bert"):
	model_name = "bert-base-uncased" #! UNcased, not cased...
	act_dim = 768
elif (model_type == "bge"):
	model_name = "BAAI/bge-large-en-v1.5"
	act_dim = 1024


device = "cuda"
seq_len = 512
embed_size = 768
batch_size = 12 #Largest number that evenly divides
total_data = 361668
total_batches = int(total_data/batch_size) #30139 = 361668/12

save_steps = [(i*10) - 1 for i in range(1,3014)]
save_steps.append(int(total_batches-1))



#? Total data size 361668,  when we divide

#endregion* Standard defs



#region* INLP (Iterative NulLspace Projection) Defs


num_classifiers = 200 #TODO How many classifiers are needed?
classifier_class = SGDClassifier #Perceptron
input_dim = embed_size
is_autoregressive = True #TODO Is_autoregressive?
min_accuracy = 0.0 #TODO min_accuracy?

#endregion* INLP (Iterative NulLspace Projection) Defs




#endregion Defs



#region Paths

root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/INLP/"

loads_p = root_p + "loads/"
saves_p = root_p + "saves/"


malefem_labs_p = loads_p + "real_malefem_inlp_labs_7_11_25.pt"

full_labs_p = loads_p + "cleargen_labs_26_9_25.pkl"
full_tokds_p = loads_p + "gentpick2_tokds.pt"

flat_np_labs_p = loads_p + "flat_np_labs.pkl"
flat_np_dat_p = loads_p + "flat_np_dat.pkl"
flat_np_tokds_p = loads_p + "flat_np_tokds.pt"

sae_p = loads_p + "pre1_sae6_7000.pt"

tpickfull_p = loads_p + "tpickfull.pkl"

#endregion Paths



#region Classes

class Classifier(object):

	def __init__(self):

		pass

	def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
		"""

		:param X_train:
		:param Y_train:
		:param X_dev:
		:param Y_dev:
		:return: accuracy score on the dev set
		"""
		raise NotImplementedError

	def get_weights(self) -> np.ndarray:
		"""
		:return: final weights of the model, as np array
		"""

		raise NotImplementedError


class SKlearnClassifier(Classifier):

	def __init__(self, m):

		self.model = m

	def train_network(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:

		"""
		:param X_train:
		:param Y_train:
		:param X_dev:
		:param Y_dev:
		:return: accuracy score on the dev set / Person's R in the case of regression
		"""

		self.model.fit(X_train, Y_train)
		score = self.model.score(X_dev, Y_dev)
		return score

	def get_weights(self) -> np.ndarray:
		"""
		:return: final weights of the model, as np array
		"""

		w = self.model.coef_
		if len(w.shape) == 1:
				w = np.expand_dims(w, 0)

		return w



#endregion Classes



#region Funcs

#? Split tensors into train/test
def split_data(data_tensor, label_tensor, split_ratio=0.8):
    """
    Splits data and label tensors into training and testing sets.

    Args:
        data_tensor (torch.Tensor): The tensor containing the features/data.
        label_tensor (torch.Tensor): The tensor containing the labels.
        split_ratio (float): The proportion of the dataset to allocate for training.
                            Defaults to 0.8 (80% train, 20% test).

    Returns:
        tuple: A tuple containing (X_train, X_test, y_train, y_test).
    """
    # Ensure the tensors have the same number of samples
    assert data_tensor.shape[0] == label_tensor.shape[0], \
        f"Data and labels must have the same number of samples, but got " \
        f"{data_tensor.shape[0]} and {label_tensor.shape[0]}"

    # Get the total number of samples
    num_samples = data_tensor.shape[0]

    # Create a random permutation of indices
    shuffled_indices = torch.randperm(num_samples)

    # Determine the split point
    train_size = int(num_samples * split_ratio)
    
    # Split the indices
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    # Create the split tensors
    X_train = data_tensor[train_indices]
    X_test = data_tensor[test_indices]
    y_train = label_tensor[train_indices]
    y_test = label_tensor[test_indices]

    return X_train, X_test, y_train, y_test

#########################################
#########################################

def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
	"""
	:param W: the matrix over its nullspace to project
	:return: the projection matrix over the rowspace
	"""

	if np.allclose(W, 0):
		w_basis = np.zeros_like(W.T)
	else:
		w_basis = scipy.linalg.orth(W.T) # orthogonal basis

	P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace

	return P_W


def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
	"""
	Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
	this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
	uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
	N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
	:param rowspace_projection_matrices: List[np.array], a list of rowspace projections
	:param dim: input dim
	"""

	I = np.eye(input_dim)
	Q = np.sum(rowspace_projection_matrices, axis = 0)
	P = I - get_rowspace_projection(Q)

	return P


def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
	"""
	the goal of this function is to perform INLP on a set of user-provided directiosn (instead of learning those directions).
	:param directions: list of vectors, as numpy arrays.
	:param input_dim: dimensionality of the vectors.
	"""

	rowspace_projections = []

	for v in directions:
		P_v = get_rowspace_projection(v)
		rowspace_projections.append(P_v)

	P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

	return P


#region? Info
#? :param classifier_class: the sklearn classifier class (SVM/Perceptron etc.)
#? :param cls_params: a dictionary, containing the params for the sklearn classifier
#? :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
#? :param input_dim: size of input vectors
#? :param is_autoregressive: whether to train the ith classiifer on the data projected to the nullsapces of w1,...,wi-1
#? :param min_accuracy: above this threshold, ignore the learned classifier
#? :param X_train: ndarray, training vectors
#? :param Y_train: ndarray, training labels (protected attributes)
#? :param X_dev: ndarray, eval vectors
#? :param Y_dev: ndarray, eval labels (protected attributes)
#? :param by_class: if true, at each iteration sample one main-task label, and extract the protected attribute only from vectors from this class
#? :param T_train_main: ndarray, main-task train labels
#? :param Y_dev_main: ndarray, main-task eval labels
#? :param dropout_rate: float, default: 0 (note: not recommended to be used with autoregressive=True)
#? :return: P, the debiasing projection; rowspace_projections, the list of all rowspace projection; Ws, the list of all calssifiers.
#endregion?
def get_debiasing_projection(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
							is_autoregressive: bool,
							min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
							Y_dev: np.ndarray, by_class=False, Y_train_main=None,
							Y_dev_main=None, dropout_rate = 0) -> np.ndarray:


	if dropout_rate > 0 and is_autoregressive:
		warnings.warn("Note: when using dropout with autoregressive training, the property w_i.dot(w_(i+1)) = 0 no longer holds.")

	I = np.eye(input_dim)

	if by_class:
		if ((Y_train_main is None) or (Y_dev_main is None)):
			raise Exception("Need main-task labels for by-class training.")
		main_task_labels = list(set(Y_train_main.tolist()))

	X_train_cp = X_train.copy()
	X_dev_cp = X_dev.copy()
	rowspace_projections = []
	Ws = []

	pbar = tqdm(range(num_classifiers))
	for i in pbar:

		clf = SKlearnClassifier(classifier_class(**cls_params))
		dropout_scale = 1./(1 - dropout_rate + 1e-6)
		dropout_mask = (np.random.rand(*X_train.shape) < (1-dropout_rate)).astype(float) * dropout_scale


		if by_class:
			#cls = np.random.choice(Y_train_main)  # uncomment for frequency-based sampling
			cls = random.choice(main_task_labels)
			relevant_idx_train = Y_train_main == cls
			relevant_idx_dev = Y_dev_main == cls
		else:
			relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
			relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

		acc = clf.train_network((X_train_cp * dropout_mask)[relevant_idx_train], Y_train[relevant_idx_train], X_dev_cp[relevant_idx_dev], Y_dev[relevant_idx_dev])
		pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
		if acc < min_accuracy: continue

		W = clf.get_weights()
		Ws.append(W)
		P_rowspace_wi = get_rowspace_projection(W) # projection to W's rowspace
		rowspace_projections.append(P_rowspace_wi)

		if is_autoregressive:

			"""
			to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far (instaed of doing X = P_iX,
			which is problematic when w_i is not exactly orthogonal to w_i-1,...,w1, due to e.g inexact argmin calculation).
			"""
			# use the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
			# N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))

			P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
			# project

			X_train_cp = (P.dot(X_train.T)).T
			X_dev_cp = (P.dot(X_dev.T)).T

	"""
	calculae final projection matrix P=PnPn-1....P2P1
	since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
	by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability and also generalize to the non-orthogonal case (e.g. with dropout),
	i.e., we explicitly project to intersection of all nullspaces (this is not critical at this point; I-(P1+...+PN) is roughly as accurate as this provided no dropout & regularization)
	"""

	P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

	return P, rowspace_projections, Ws


#endregion Funcs



#region Model

model = LanguageModel(model_name,device_map=device)
model.eval() #No changing weights.

submodule_ref6 = eval("model.bert.encoder.layer[6]") 
submodule_ref7 = eval("model.bert.encoder.layer[7]")
submodule_ref_last = eval("model.bert.encoder.layer[11]")

#endregion Model



#region Tokenizer

tok = model.tokenizer
tok.add_special_tokens({'pad_token': '[PAD]'})
tok.backend_tokenizer.enable_truncation(max_length=seq_len)
tok.backend_tokenizer.enable_padding(length=seq_len, pad_id=tok.pad_token_id,pad_token=tok.pad_token)

#endregion Tokenizer



#region Datasets


#? No need to for full ones for now
# with open(tpickfull_p,"rb") as f:
# 	tpickfull = pickle.load(f)

#? No need to for full ones for now
with open(full_labs_p,"rb") as f:
	full_labs = pickle.load(f)

#? Already created the act data, no logner needed
# with open(full_tokds_p,"rb") as f:
# 	full_tokds = torch.load(f,weights_only=False)



with open(malefem_labs_p,"rb") as f:
	malefem_labs= torch.load(f)


with open(flat_np_labs_p,"rb") as f:
	flat_np_labs= pickle.load(f)


#? Already created the act data, no logner needed
# with open(flat_np_dat_p,"rb") as f:
# 	flat_np_dat= pickle.load(f)

#? Already created the act data, no logner needed
# with open(flat_np_tokds_p,"rb") as f:
# 	flat_np_tokds = torch.load(f,weights_only=False)



# loader = torch.utils.data.DataLoader(
# 	flat_np_tokds,
# 	batch_size=batch_size,
# 	num_workers=4, #Start at this value, later we can move higher up - generally up to number of cores.
# 	prefetch_factor=8, #Decent to start here - how many batches are loaded at once
# 	persistent_workers=True,

# )


# diter = iter(loader)



#endregion Datasets



#region# Transforming inputs (Labs and Acts)

# allchunk_labs_l = []

# sorted_labs = [[labs[j]]*len(tpickfull[j]) for j in range(len(labs))]

# for l in sorted_labs:
#     allchunk_labs_l += l

# allchunk_labs = torch.stack(allchunk_labs_l) #? torch.stack, not cat.

# labs_fin = allchunk_labs.numpy()

# labs_fin =labs_fin.astype(int)

# allchunk_dat = torch.cat(tpickfull)

# dat_fin = allchunk_dat.numpy()


#endregion#


#region# Get hidden_acts


#region*# Compat test

# stoppedat = 27559 #? Used for manual batching; There are better ways of doing this.
# for i in range(0,int(stoppedat)):
# 	keeplast = next(diter)

# old_i = [7619,7629,7639,]

# old_p = [f"/media/strah/344A08F64A08B720/Work_related/INLP_saves/nullspace_acts{i}.pt" for i in old_i]

# acts_old = []

# for p in old_p:
# 	with open(p,"rb") as f:
# 		acts_old.append(torch.load(f))

# #########################################
# #########################################
# #########################################

# new_i = [7619,7629,7639,7649,7659]

# new_p = [f"/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/INLP/saves/nullspace_acts{i}.pt" for i in new_i]

# acts_new = []

# for p in new_p:
# 	with open(p,"rb") as f:
# 		acts_new.append(torch.load(f))


# print("STOP")
# print("STOP")
#endregion*#


# save_list = []


# with tqdm(total=total_batches,desc="Gathering activations...") as loop:
# 	loop.n = stoppedat
# 	for batch_idx in range(loop.n,total_batches):

		#region*# act inputs

# 		batch = next(diter)
# 		token_type_ids = torch.zeros(batch[0].size())
# 		#Reconstructing the BatchEncoding object's dictionary manually.
# 		inputs = {"input_ids":batch[0].type(torch.int64).to(device),"token_type_ids":token_type_ids,"attention_mask":batch[1]}


		#endregion*#


		#region*# Extraction

# 		with torch.inference_mode():
# 			with model.trace(inputs) as tracer:
			
# 				out6 = submodule_ref6.nns_output[0].save()


# 			save_list.append(out6.to("cpu",non_blocking=True))
			


		#endregion*#


# 		region#* Saving!
# 		#! OOM at 7649
# 		if(loop.n in save_steps):

			

# 			sp = saves_p + f"nullspace_acts{loop.n}.pt"
		
# 			to_save = torch.cat(save_list,dim=0) #First dimension is the batch dimension


# 			with open(sp,"wb") as f:
# 				torch.save(to_save,f)


# 			while save_list:
# 				del save_list[0]
# 			gc.collect()


# 		if(loop.n == 30139):

# 			print("STOP")
# 			print("STOP")

		#endregion*#


# 		loop.update(1)




#endregion#


#region Process acts for training 


#region*# Old; Loading direct acts

# acts_lp = "/media/strah/344A08F64A08B720/Work_related/INLP_saves/"

# actidx_from = 270
# actidx_to = 300


# allact_indies = [(i*10)-1 for i in range(1,3014)]
# actsel_indies = allact_indies[actidx_from:actidx_to]


# act_list = []


# for idx in tqdm(actsel_indies,desc="Loading acts for dataset..."):
# 	p = acts_lp + f"nullspace_acts{idx}.pt"

# 	with open(p,"rb") as f:
# 		act_list.append(torch.load(f))


# act_t  = torch.cat(act_list,dim=0)


# #? These are determined from the actidx by *120, since there are 120 entries per act 
# labidx_from = actidx_from*120
# labidx_to = actidx_to*120

# labs_t = torch.Tensor(flat_np_labs[labidx_from:labidx_to])

# labs_t = labs_t.unsqueeze(-1).expand(-1,512).reshape([-1])


#endregion*#

#? Use subset of data
og_max_per_class = 6386
mf_max_per_class = 8401
sample_per_class = 500


#region* Balancing data

nzs = (full_labs == 1).nonzero(as_tuple=True)[0]
nzs_size = nzs.size()
nzs = nzs[torch.randperm(len(nzs))][:sample_per_class]
zs = (full_labs == 0).nonzero(as_tuple=True)


mf_zeros = (malefem_labs == 0).nonzero(as_tuple=True)[0]
zeros_size = mf_zeros.size()
mf_zeros = mf_zeros[torch.randperm(len(mf_zeros))][:sample_per_class]
mf_ones = (malefem_labs == 1).nonzero(as_tuple=True)


#? Create random permutation of idxs for negatives, and only take the amount to equal positives
zs_r = zs[0][torch.randperm(len(zs[0]))][:sample_per_class]
mf_ones_r = mf_ones[0][torch.randperm(len(mf_ones[0]))][:sample_per_class]

idxs = torch.cat([nzs,zs_r])
mf_idxs = torch.cat([mf_zeros,mf_ones_r])

idxs_r = idxs[torch.randperm(len(idxs))]
mf_idxs_r = mf_idxs[torch.randperm(len(mf_idxs))]


#region* Loading Relevant acts

avg_act_p = loads_p + "avged_acts/"
act_list = []
# for i in idxs_r.tolist():
for i in mf_idxs_r.tolist():
	act_list.append(torch.load(avg_act_p+f"averaged_doc{i}.pt"))
act_t = torch.stack(act_list)

#endregion* Loading Relevant acts

labs_t = full_labs[idxs_r]
mf_labs_t = malefem_labs[mf_idxs_r] #TODO Need to do acts with mf indexes too for them to line up

#endregion* Balancing data


labs_t = labs_t.unsqueeze(-1).expand(-1,512).reshape([-1])
mf_labs_t = mf_labs_t.unsqueeze(-1).expand(-1,512).reshape([-1])


act_t = act_t.view([-1,768]) #? Make sure to resize to 2d

print("STOP")
print("STOP")

#? Free the memory from the uncollated acts
while act_list:
	del act_list[0]
gc.collect()


#endregion Process acts for training



#region curr Testing OG code



#region* Generating Random input data

# N = 10000
# d = 300
# X = np.random.rand(N, d) - 0.5 #? Creates a [10000,300] input
# Y = np.array([1 if sum(x) > 0 else 0 for x in X]) #? If a row sum is > 0 (after the -0.5) then 1.

#X < 0 
# #np.random.rand(N) < 0.5  #? 0.1: Direct linear signal, uses X to generate labels
# #(X + 0.01 * (np.random.rand(*X.shape) - 0.5)) < 0  #? 0.2: Noisy linear signal
# #np.random.rand(5000) < 0.5 #? 0.3: Random signal
#Y = np.array(Y, dtype = int) #? Turns one of the above into a Y

#endregion* Generating Random input data


#region* Defs
num_classifiers = 100
classifier_class = SGDClassifier #Perceptron...
input_dim = embed_size 
is_autoregressive = True
min_accuracy = 0.0
#endregion* Defs

act_np = act_t.numpy()

# labs_np = labs_t.numpy()
labs_np = mf_labs_t.numpy()


x_trn,x_val,y_trn,y_val = train_test_split(act_np,labs_np,test_size=0.2,random_state=42,stratify=labs_np)


# P, rowspace_projections, Ws = get_debiasing_projection(
#     classifier_class,
#     {},
#     num_classifiers,
#     input_dim,
#     is_autoregressive,
#     min_accuracy,
#     x_trn,
#     y_trn,
#     x_val, #? In example, they used the same exact one for train and valid
#     y_val,
#     by_class = False,)


with open(saves_p+"first_step.pkl","rb") as f:
	P, rowspace_projections, Ws = pickle.load(f)



# print("STOP")
# print("STOP")

I = np.eye(P.shape[0])
P_alternative = I - np.sum(rowspace_projections, axis = 0)
P_by_product = I.copy()

for P_Rwi in rowspace_projections:

    P_Nwi = I - P_Rwi
    P_by_product = P_Nwi.dot(P_by_product)

fin_save_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/INLP/saves/fin_step.pt"

with open(fin_save_p,"wb") as f:
	torch.save(P_by_product,f)

print("STOP")
print("STOP")

#endregion curr Testing OG code


#region Main Code






#endregion Main Code





print("STOP")
print("STOP")


print("END FILE")
