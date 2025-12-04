from tqdm import tqdm
import os
import torch
import pickle
from collections import namedtuple
from functools import partial
import random

from datasets import load_dataset

from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from nnsight import LanguageModel

from sae_lens import SAE
from sae_lens.sae import SAEConfig
from transformer_lens import HookedEncoder, HookedTransformer

#TODO probs remove this.
import sae_bench.custom_saes.identity_sae as identity_sae

import time
from sae_bench.custom_saes.relu_sae import ReluSAE
from sae_bench.custom_saes import custom_sae_config
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.sae_bench_utils import activation_collection

import sae_bench.sae_bench_utils.general_utils as general_utils
from torch import Tensor
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypeAlias
from collections.abc import Iterator
from openai import OpenAI
from sae_bench.sae_bench_utils.indexing_utils import (
    get_iw_sample_indices,
    get_k_largest_indices,
    index_with_buffer,
)


#* imports for debug run_eval
#region

EVAL_TYPE_ID_AUTOINTERP = "autointerp"
from sae_bench.evals.autointerp.main import run_eval, AutoInterp
from sae_bench.sae_bench_utils.activation_collection import get_feature_activation_sparsity
from typing import Any
from sae_bench.sae_bench_utils import get_eval_uuid,get_sae_bench_version,get_sae_lens_version
from sae_bench.evals.autointerp.main import run_eval_single_sae
from sae_bench.evals.autointerp.eval_output import AutoInterpEvalOutput, AutoInterpMetricCategories, AutoInterpMetrics
import sae_bench.sae_bench_utils.dataset_utils as dataset_utils
from tabulate import tabulate
import datetime
import gc
from dataclasses import asdict
import asyncio

#endregion  


#* Run params
#region

#Are we loading or processing the sparse activation tensors
ld_spars_tens = True

#endregion  



##############################

Messages: TypeAlias = list[dict[Literal["role", "content"], str]]

class Example:
    """
    Data for a single example sequence.
    """

    def __init__(
        self,
        toks: list[int],
        acts: list[float],
        act_threshold: float,
        model,
    ):
        self.toks = toks
        # self.str_toks = model.to_str_tokens(torch.tensor(self.toks)) #Using below as HookedEncoder does not have this
        self.str_toks = model.tokenizer.convert_ids_to_tokens(torch.tensor(self.toks))
        self.acts = acts
        self.act_threshold = act_threshold
        self.toks_are_active = [act > act_threshold for act in self.acts]
        self.is_active = any(
            self.toks_are_active
        )  # this is what we predict in the scoring phase

    def to_str(self, mark_toks: bool = False) -> str:
        return (
            "".join(
                f"<<{tok}>>" if (mark_toks and is_active) else tok
                for tok, is_active in zip(self.str_toks, self.toks_are_active)  # type: ignore
            )
            .replace("�", "")
            .replace("\n", "↵")
            # .replace(">><<", "")
        )


def str_bool(b: bool) -> str:
    return "Y" if b else ""


class Examples:
    """
    Data for multiple example sequences. Includes methods for shuffling seuqences, and displaying them.
    """

    def __init__(self, examples: list[Example], shuffle: bool = False) -> None:
        self.examples = examples
        if shuffle:
            random.shuffle(self.examples)
        else:
            self.examples = sorted(
                self.examples, key=lambda x: max(x.acts), reverse=True
            )

    def display(self, predictions: list[int] | None = None) -> str:
        """
        Displays the list of sequences. If `predictions` is provided, then it'll include a column for both "is_active"
        and these predictions of whether it's active. If not, then neither of those columns will be included.
        """
        return tabulate(
            [
                (
                    [max(ex.acts), ex.to_str(mark_toks=True)]
                    if predictions is None
                    else [
                        max(ex.acts),
                        str_bool(ex.is_active),
                        str_bool(i + 1 in predictions),
                        ex.to_str(mark_toks=False),
                    ]
                )
                for i, ex in enumerate(self.examples)
            ],
            headers=["Top act"]
            + ([] if predictions is None else ["Active?", "Predicted?"])
            + ["Sequence"],
            tablefmt="simple_outline",
            floatfmt=".3f",
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


class AutoInterp:
    """
    This is a start-to-end class for generating explanations and optionally scores. It's easiest to implement it as a
    single class for the time being because there's data we'll need to fetch that'll be used in both the generation and
    scoring phases.
    """

    def __init__(
        self,
        cfg: AutoInterpEvalConfig,
        model: HookedTransformer,
        sae: SAE,
        tokenized_dataset: Tensor,
        sparsity: Tensor,
        device: str,
        api_key: str,
        is_bert = False, #!Strah added
    ):
        self.cfg = cfg
        self.model = model
        self.sae = sae
        self.tokenized_dataset = tokenized_dataset
        self.device = device
        self.api_key = api_key
        if cfg.latents is not None:
            self.latents = cfg.latents
        else:
            assert self.cfg.n_latents is not None
            sparsity *= cfg.total_tokens
            alive_latents = (
                torch.nonzero(sparsity > self.cfg.dead_latent_threshold)
                .squeeze(1)
                .tolist()
            )
            if len(alive_latents) < self.cfg.n_latents:
                self.latents = alive_latents
                print(
                    f"\n\n\nWARNING: Found only {len(alive_latents)} alive latents, which is less than {self.cfg.n_latents}\n\n\n"
                )
            else:
                self.latents = random.sample(alive_latents, k=self.cfg.n_latents)
        self.n_latents = len(self.latents)
        self.is_bert = is_bert

    async def run(
        self, 
        explanations_override: dict[int, str] = {}
    ) -> dict[int, dict[str, Any]]:
        """
        Runs both generation & scoring phases. Returns a dict where keys are latent indices, and values are dicts with:

            "explanation": str, the explanation generated for this latent
            "predictions": list[int], the predicted activating indices
            "correct seqs": list[int], the true activating indices
            "score": float, the fraction of correct predictions (including positive and negative)
            "logs": str, the logs for this latent
        """

        #Use to be my stuff
        generation_examples, scoring_examples = self.gather_data()


        latents_with_data = sorted(generation_examples.keys())
        n_dead = self.n_latents - len(latents_with_data)
        if n_dead > 0:
            print(
                f"Found data for {len(latents_with_data)}/{self.n_latents} alive latents; {n_dead} dead"
            )

        with ThreadPoolExecutor(max_workers=2) as executor: #TODO Find the right amount.
            tasks = [
                self.run_single_feature(
                    executor,
                    latent,
                    generation_examples[latent],
                    scoring_examples[latent],
                    explanations_override.get(latent, None),
                )
                for latent in latents_with_data
            ]
            results = {}
            for future in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Calling API (for gen & scoring)",
            ):
                result = await future
                if result:
                    results[result["latent"]] = result

        return results

    async def run_single_feature(
        self,
        executor: ThreadPoolExecutor,
        latent: int,
        generation_examples: Examples,
        scoring_examples: Examples,
        explanation_override: str | None = None,
    ) -> dict[str, Any] | None:
        # Generation phase
        gen_prompts = self.get_generation_prompts(generation_examples)
        (explanation_raw,), logs = await asyncio.get_event_loop().run_in_executor(
            executor,
            self.get_api_response,
            gen_prompts,
            self.cfg.max_tokens_in_explanation,
        )
        explanation = self.parse_explanation(explanation_raw)
        results = {
            "latent": latent,
            "explanation": explanation,
            "logs": f"Generation phase\n{logs}\n{generation_examples.display()}",
        }

        # Scoring phase
        if self.cfg.scoring:
            scoring_prompts = self.get_scoring_prompts(
                explanation=explanation_override or explanation,
                scoring_examples=scoring_examples,
            )
            (predictions_raw,), logs = await asyncio.get_event_loop().run_in_executor(
                executor,
                self.get_api_response,
                scoring_prompts,
                self.cfg.max_tokens_in_prediction,
            )
            predictions = self.parse_predictions(predictions_raw)
            if predictions is None:
                return None
            score = self.score_predictions(predictions, scoring_examples)
            results |= {
                "predictions": predictions,
                "correct seqs": [
                    i for i, ex in enumerate(scoring_examples, start=1) if ex.is_active
                ],
                "score": score,
                "logs": results["logs"]
                + f"\nScoring phase\n{logs}\n{scoring_examples.display(predictions)}",
            }

        return results

    def parse_explanation(self, explanation: str) -> str:
        return explanation.split("activates on")[-1].rstrip(".").strip()

    def parse_predictions(self, predictions: str) -> list[int] | None:
        predictions_split = (
            predictions.strip()
            .rstrip(".")
            .replace("and", ",")
            .replace("None", "")
            .split(",")
        )
        predictions_list = [i.strip() for i in predictions_split if i.strip() != ""]
        if predictions_list == []:
            return []
        if not all(pred.strip().isdigit() for pred in predictions_list):
            return None
        predictions_ints = [int(pred.strip()) for pred in predictions_list]
        return predictions_ints

    def score_predictions(
        self, predictions: list[int], scoring_examples: Examples
    ) -> float:
        classifications = [
            i in predictions for i in range(1, len(scoring_examples) + 1)
        ]
        correct_classifications = [ex.is_active for ex in scoring_examples]
        return sum(
            [c == cc for c, cc in zip(classifications, correct_classifications)]
        ) / len(classifications)

    def get_api_response(
        self, messages: Messages, max_tokens: int, n_completions: int = 1
    ) -> tuple[list[str], str]:
        """Generic API usage function for OpenAI"""
        for message in messages:
            assert message.keys() == {"content", "role"}
            assert message["role"] in ["system", "user", "assistant"]

        client = OpenAI(api_key=self.api_key)

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # type: ignore
            n=n_completions,
            max_tokens=max_tokens,
            stream=False,
        )
        response = [choice.message.content.strip() for choice in result.choices]

        logs = tabulate(
            [
                m.values()
                for m in messages + [{"role": "assistant", "content": response[0]}]
            ],
            tablefmt="simple_grid",
            maxcolwidths=[None, 120],
        )

        return response, logs

    def get_generation_prompts(self, generation_examples: Examples) -> Messages:
        assert len(generation_examples) > 0, "No generation examples found"

        examples_as_str = "\n".join(
            [
                f"{i + 1}. {ex.to_str(mark_toks=True)}"
                for i, ex in enumerate(generation_examples)
            ]
        )

        SYSTEM_PROMPT = """We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the neuron activates, in order from most strongly activating to least strongly activating. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try not to be overly specific in your explanation. Note that some neurons will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the neuron to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words."""
        if self.cfg.use_demos_in_explanation:
            SYSTEM_PROMPT += """ Some examples: "This neuron activates on the word 'knows' in rhetorical questions", and "This neuron activates on verbs related to decision-making and preferences", and "This neuron activates on the substring 'Ent' at the start of words", and "This neuron activates on text about government economic policy"."""
        else:
            SYSTEM_PROMPT += (
                """Your response should be in the form "This neuron activates on..."."""
            )
        USER_PROMPT = (
            f"""The activating documents are given below:\n\n{examples_as_str}"""
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def get_scoring_prompts(
        self, explanation: str, scoring_examples: Examples
    ) -> Messages:
        assert len(scoring_examples) > 0, "No scoring examples found"

        examples_as_str = "\n".join(
            [
                f"{i + 1}. {ex.to_str(mark_toks=False)}"
                for i, ex in enumerate(scoring_examples)
            ]
        )

        example_response = sorted(
            random.sample(
                range(1, 1 + self.cfg.n_ex_for_scoring),
                k=self.cfg.n_correct_for_scoring,
            )
        )
        example_response_str = ", ".join([str(i) for i in example_response])
        SYSTEM_PROMPT = f"""We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this neuron activates for, and then be shown {self.cfg.n_ex_for_scoring} example sequences in random order. You will have to return a comma-separated list of the examples where you think the neuron should activate at least once, on ANY of the words or substrings in the document. For example, your response might look like "{example_response_str}". Try not to be overly specific in your interpretation of the explanation. If you think there are no examples where the neuron will activate, you should just respond with "None". You should include nothing else in your response other than comma-separated numbers or the word "None" - this is important."""
        USER_PROMPT = f"Here is the explanation: this neuron fires on {explanation}.\n\nHere are the examples:\n\n{examples_as_str}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def gather_data(
        self,
        ) -> tuple[dict[int, Examples], dict[int, Examples]]:
        """
        Stores top acts / random seqs data, which is used for generation & scoring respectively.
        """
        dataset_size, seq_len = self.tokenized_dataset.shape

        #!##########################################
        #?Strah added this here - not in official release of the library.

        if (self.is_bert):
            acts = activation_collection.collect_sae_activations_for_bert(
                self.tokenized_dataset,
                self.model,
                self.model.tokenizer,
                self.sae,
                self.cfg.llm_batch_size,
                self.sae.cfg.hook_layer,
                self.sae.cfg.hook_name,
                mask_bos_pad_eos_tokens=True,
                selected_latents=self.latents,
                activation_dtype=torch.float32,

            )

        else:
            acts = activation_collection.collect_sae_activations(
                self.tokenized_dataset,
                self.model,
                self.sae,
                self.cfg.llm_batch_size,
                self.sae.cfg.hook_layer,
                self.sae.cfg.hook_name,
                mask_bos_pad_eos_tokens=True,
                selected_latents=self.latents,
                activation_dtype=torch.bfloat16,  # reduce memory usage, we don't need full precision when sampling activations
            )

        #!##########################################

        generation_examples = {}
        scoring_examples = {}

        for i, latent in tqdm(
            enumerate(self.latents), desc="Collecting examples for LLM judge"
        ):
            # (1/3) Get random examples (we don't need their values)
            rand_indices = torch.stack(
                [
                    torch.randint(0, dataset_size, (self.cfg.n_random_ex_for_scoring,)),
                    torch.randint(
                        self.cfg.buffer,
                        seq_len - self.cfg.buffer,
                        (self.cfg.n_random_ex_for_scoring,),
                    ),
                ],
                dim=-1,
            ).to(self.tokenized_dataset.device)
            rand_toks = index_with_buffer(
                self.tokenized_dataset, rand_indices, buffer=self.cfg.buffer
            )

            # (2/3) Get top-scoring examples
            top_indices = get_k_largest_indices(
                acts[..., i],
                k=self.cfg.n_top_ex,
                buffer=self.cfg.buffer,
                no_overlap=self.cfg.no_overlap,
            )
            top_toks = index_with_buffer(
                self.tokenized_dataset, top_indices, buffer=self.cfg.buffer
            )
            top_values = index_with_buffer(
                acts[..., i], top_indices, buffer=self.cfg.buffer
            )
            act_threshold = self.cfg.act_threshold_frac * top_values.max().item()

            # (3/3) Get importance-weighted examples, using a threshold so they're disjoint from top examples
            # Also, if we don't have enough values, then we assume this is a dead feature & continue
            threshold = top_values[:, self.cfg.buffer].min().item()
            acts_thresholded = torch.where(acts[..., i] >= threshold, 0.0, acts[..., i])
            if acts_thresholded[:, self.cfg.buffer : -self.cfg.buffer].max() < 1e-6:
                continue
            iw_indices = get_iw_sample_indices(
                acts_thresholded, k=self.cfg.n_iw_sampled_ex, buffer=self.cfg.buffer
            )
            iw_toks = index_with_buffer(
                self.tokenized_dataset, iw_indices, buffer=self.cfg.buffer
            )
            iw_values = index_with_buffer(
                acts[..., i], iw_indices, buffer=self.cfg.buffer
            )

            # Get random values to use for splitting
            rand_top_ex_split_indices = torch.randperm(self.cfg.n_top_ex)
            top_gen_indices = rand_top_ex_split_indices[
                : self.cfg.n_top_ex_for_generation
            ]
            top_scoring_indices = rand_top_ex_split_indices[
                self.cfg.n_top_ex_for_generation :
            ]
            rand_iw_split_indices = torch.randperm(self.cfg.n_iw_sampled_ex)
            iw_gen_indices = rand_iw_split_indices[
                : self.cfg.n_iw_sampled_ex_for_generation
            ]
            iw_scoring_indices = rand_iw_split_indices[
                self.cfg.n_iw_sampled_ex_for_generation :
            ]

            def create_examples(
                all_toks: Tensor, all_acts: Tensor | None = None
            ) -> list[Example]:
                if all_acts is None:
                    all_acts = torch.zeros_like(all_toks).float()
                return [
                    Example(
                        toks=toks,
                        acts=acts,
                        act_threshold=act_threshold,
                        model=self.model,
                    )
                    for (toks, acts) in zip(all_toks.tolist(), all_acts.tolist())
                ]

            # Get the generation & scoring examples
            generation_examples[latent] = Examples(
                create_examples(top_toks[top_gen_indices], top_values[top_gen_indices])
                + create_examples(iw_toks[iw_gen_indices], iw_values[iw_gen_indices]),
            )
            scoring_examples[latent] = Examples(
                create_examples(
                    top_toks[top_scoring_indices], top_values[top_scoring_indices]
                )
                + create_examples(
                    iw_toks[iw_scoring_indices], iw_values[iw_scoring_indices]
                )
                + create_examples(rand_toks),
                shuffle=True,
            )

        return generation_examples, scoring_examples



#####################









#* Paths
#region

#for dataloader
data_path= "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/"
tensor_tok_lex_p = data_path + "Asylex_Dataset/"+ "tensored_tok_lex.pkl"

root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/tester/autointerp/"

param_path = root_p + "ae.pt" #aka sae path
output_folder = root_p + "outs/"
res_p = output_folder + "results.pkl"



#endregion  


#* Funcs and classes
#region

#Used to create a data loader from the Asylex dataset
class AsyLexDataset(torch.utils.data.Dataset):

	def __init__(self, data):
		# super().__init__()
		self.data = data
	

	def __len__(self):
		return len(self.data)


	def __getitem__(self, i):
		return {"text":self.data[i]}


#Adapted from relu_sae.py
def load_my_relu_sae(
    param_path: str,
    config: dict,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: int | None = None,
) -> ReluSAE:


    pt_params = torch.load(param_path, map_location=torch.device("cpu"))



    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = ReluSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        model_name=model_name,
        hook_layer=layer,  # type: ignore
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(renamed_params)  # type: ignore


    sae.to(device=device, dtype=dtype)



    d_sae, d_in = sae.W_dec.data.shape
    assert d_sae >= d_in



    if config["trainer"].wandb_name == "StandardTrainer":
        sae.cfg.architecture = "standard"
    elif config["trainer"].wandb_name == "PAnnealTrainer":
        sae.cfg.architecture = "p_anneal"
    elif config["trainer"].wandb_name == "StandardTrainerAprilUpdate":
        sae.cfg.architecture = "standard_april_update"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer'].wandb_name}")

    normalized = sae.check_decoder_norms()
    if not normalized:
        sae.normalize_decoder()


    return sae

def escape_slash(s: str) -> str:
    return s.replace("/", "_")


def run_eval_single_sae(
    config: AutoInterpEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    device: str,
    artifacts_folder: str,
    api_key: str,
    sae_sparsity: torch.Tensor | None = None,
) -> dict[str, float]:
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.set_grad_enabled(False)

    os.makedirs(artifacts_folder, exist_ok=True)

    tokens_filename = f"{escape_slash(config.model_name)}_{config.total_tokens}_tokens_{config.llm_context_size}_ctx.pt"
    tokens_path = os.path.join(artifacts_folder, tokens_filename)

    if os.path.exists(tokens_path):
        tokenized_dataset = torch.load(tokens_path).to(device)
    else:
        tokenized_dataset = dataset_utils.load_and_tokenize_dataset(
            config.dataset_name,
            config.llm_context_size,
            config.total_tokens,
            model.tokenizer,  # type: ignore
        ).to(device)
        torch.save(tokenized_dataset, tokens_path)

    print(f"Loaded tokenized dataset of shape {tokenized_dataset.shape}")

    if sae_sparsity is None:
        sae_sparsity = activation_collection.get_feature_activation_sparsity(
            tokenized_dataset,
            model,
            sae,
            config.llm_batch_size,
            sae.cfg.hook_layer,
            sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
        )

    autointerp = AutoInterp(
        cfg=config,
        model=model,
        sae=sae,
        tokenized_dataset=tokenized_dataset,
        sparsity=sae_sparsity,
        api_key=api_key,
        device=device,
    )
    results = asyncio.run(autointerp.run())
    return results  # type: ignore



#*Monkey patching torch.load to set weights_only to off.
#region

og_torch_load = torch.load

def ntorch_load(*args, **kwargs):
	kwargs['weights_only'] = False
	return og_torch_load(*args, **kwargs)

torch.load = ntorch_load #?its good practice to restore the function when done

#endregion




#endregion  


#* Defs
#region


#Actively tunned
per_grp = 500
let_max = 6121
lat_grp = [[i for i in range(i,i+per_grp)] for i in range(0,let_max,per_grp)]
select_lat_grp = 0 #up to 12





#General
model_name = "bert-base-cased"
device = "cuda"
steps = 10_000
random_seed = 42
# steps = 180 #? If finetuned


#? sae defs
    #region

exp_factor = 8
act_dim = 768
dict_size = exp_factor * act_dim
dtype = torch.float32
hook_layer = 6


#Defining the actual SAE
sae_cfg= dict(
                dict_class=AutoEncoder,
				activation_dim=act_dim,
				dict_size=exp_factor * act_dim ,
				lr=1e-4, #Update1 - at 1e-3 was irratic, had to reduce
				# l1_penalty = 1e-4, #Default value 1e-1
				l1_penalty = 1.6e0, #Default value 1e-1
				device=device,
				steps=steps,
				layer=3,
				lm_name=model_name,
				warmup_steps=100, #About 1-10% of total steps as a heuristic. Try this first, see later
				sparsity_warmup_steps=100, #Shall assume the same holds for as for the above
)

#? This seems like a very strange way of doing things...
trn = StandardTrainer(**sae_cfg)
sae_cfg["trainer"] = trn

lensed_ae = load_my_relu_sae(param_path=param_path,config=sae_cfg,model_name=model_name,device=device,dtype=dtype,layer=hook_layer)

    #endregion

#? hook_bert init
    #region


hook_bert = HookedEncoder.from_pretrained(model_name).to(device)
tok = hook_bert.tokenizer

    #endregion


#? Dataset and loader
    #region



batch_size = 64
batch_num = 125 #TODO Lower even more if need be
seq_len = 512 #BERTs maximum sequence len

#?Loading in a tokenized version of the dataset
with open(tensor_tok_lex_p,"rb") as f:
    lex = pickle.load(f)
    # lex_to_ds = list(lex["raw_txt"])

lex = lex.type(torch.int).to(device)
#Using smaller dataset because of memory
lex = lex[:(batch_size*batch_num)]


#TODO try to add lex directly into the funciton first.
# dataset = AsyLexDataset(lex_to_ds)

# loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         num_workers=4, #Start at this value, later we can move higher up - generally up to number of cores.
#         prefetch_factor=8, #Decent to start here - how many batches are loaded at once
#         persistent_workers=True,
#         drop_last=True,
#     )




    #endregion


#endregion



#* Main
#region


start_time = time.time() #Start logging time


#? get the api key - this will probably be necessary whatever happens
with open(root_p+"openai_api_key.txt", "r") as f:
    api_key = f.read().strip()

hook_namer = "blocks.6.hook_resid_post"

# selected_saes = [
#     (model_name,"blocks.3.hook_resid_post"),
#     (model_name,"blocks.6.hook_resid_post"),
#     (model_name,"blocks.9.hook_resid_post"),
#     ]


# #TODO Check to see if this works.
config = AutoInterpEvalConfig(
    random_seed=random_seed,
    model_name=model_name,
    llm_batch_size=batch_size,
    llm_context_size=seq_len,
    llm_dtype="float32",
    # n_latents=6144, #Instead of random lats, we define our own
    override_latents=lat_grp[select_lat_grp]
)



# # run the evaluation on all selected SAEs
# results_dict = run_eval(
#     config,
#     selected_saes,
#     device,
#     api_key,
#     output_folder,
#     force_rerun=True,
# )



# act_spars_tens = get_feature_activation_sparsity(lex,hook_bert,lensed_ae,batch_size,hook_layer,hook_namer)





if(ld_spars_tens):
    p = root_p + "spartens_1.pt"

    with open(p,"rb") as f:
        act_spars_tens = torch.load(f)

else:

    act_spars_tens = activation_collection.get_feature_activation_sparsity_for_bert_from_tensor(
        lex,
        hook_bert,
        tok,
        lensed_ae,
        batch_size,
        hook_namer,
    )




autointerp = AutoInterp(
    cfg=config,
    model=hook_bert,
    sae=lensed_ae,
    tokenized_dataset=lex,
    sparsity=act_spars_tens,
    api_key=api_key,
    device=device,
    is_bert=True,
)


print("STOP")
print("STOP")

results = asyncio.run(autointerp.run())


print("STOP")
print("STOP")


with open(res_p,"wb") as f:
    pickle.dump(results,f)


end_time = time.time() # End logging time
print(f"Finished evaluation in {end_time - start_time} seconds")


print("STOP")
print("STOP")


#endregion


