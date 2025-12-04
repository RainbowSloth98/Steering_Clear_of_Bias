from pathlib import Path
import torch
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import run_eval
from sae_lens import SAE
from sae_lens.sae import SAEConfig




#* Paths
#region

data_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/autointerp/"

api_p = data_p + "openai_api_key.txt"
log_p = data_p + "logs/"
out_p = data_p + "outs/"
sae_p = data_p + "SAEs/SAE6_saves/"


save_logs_path = log_p + "logs.txt"
# save_logs_path.unlink(missing_ok=True)

output_path = out_p + "data"
# output_path.mkdir(exist_ok=True)


#endregion  


#* Params
#region

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model_name = "bert-base-cased"
exp_factor = 8
act_dim = 768
dtype = "float32" #as per hook_bert.cfg.dtype
batch_size = 128 #as per SAE_trainer
latent_dim = exp_factor*act_dim


selected_saes = [

    # ("gpt2-small-res-jb", "blocks.7.hook_resid_pre"),
    (model_name,"blocks.3.hook_resid_post"),
    (model_name,"blocks.6.hook_resid_post"),
    (model_name,"blocks.9.hook_resid_post"),
    
    ]


#endregion  



#*Main
#region

sae_cfg = SAEConfig(
    architecture = "standard",
    d_in = 768,
    d_sae = 6144, #8 times exp_factor
    activation_fn_str = "relu",
    apply_b_dec_to_input = False,
    finetuning_scaling_factor = False,
    context_size = 512, #seq_len
    model_name = "bert-base-cased",
    hook_name = "blocks.6.hook_resid_post", #? change for 3 and 9
    hook_layer = 6, #same as above,
    hook_head_index = None,
    prepend_bos = False, #TODO double check this
    dataset_path = "monology/pile-uncopyrighted",
    dataset_trust_remote_code = False, #TODO mb True?
    normalize_activations = "none", #TODO This might be a pain point?
    dtype = "float32",
    device = "cuda",
    sae_lens_training_version =  None,
    activation_fn_kwargs = {},
    neuronpedia_id = None,
    model_from_pretrained_kwargs = {},
    )

#*My custom attempt at the cfg.
AIEC_cfg = AutoInterpEvalConfig(

	model_name=model_name,
	device = "cuda",
	n_latents = 6144,
	override_latents = None,
	dead_latent_threshold = 4, #check
	seed = 42,
	buffer = 10, #Set to default, check other vals
	no_overlap = True, #Test false?
	act_threshold_frac = 0.01, #seems reasonable
	total_tokens = 2_000_000, #*Default, change later probs
	batch_size = 64, #Mb 128 if there isn't much overhead
	scoring = True, #Double check what is
	max_tokens_in_explanation = 30, #set to default - try more
	use_demos_in_explanation = True, #Try, if not set False
	n_top_ex_for_generation = 15, #Default is 10...mb change?
	n_iw_sampled_ex_for_generation = 5, #idk, keep default
	n_top_ex_for_scoring = 3, #default 2, try if need.
	n_random_ex_for_scoring = 10, #Keep default
	n_iw_sampled_ex_for_scoring = 2, #Keep default

)



#Apparently they use ChatGPT in the background...
with open(api_p) as f:
    api_key = f.read().strip()

torch.set_grad_enabled(False) #Same as torch.no_grad()



#TODO Fix or replace such that we can run autointerp
cfg = AutoInterpEvalConfig(

    model_name=model_name, 
    n_latents=latent_dim, 
    llm_dtype=dtype, 
    llm_batch_size=batch_size,
)


results = run_eval(
    cfg,
    selected_saes,
    str(device),
    api_key,
    output_path=str(output_path),
    save_logs_path=str(save_logs_path),
)  # type: ignore


print(results)

print("STOP")
print("STOP")


#endregion

