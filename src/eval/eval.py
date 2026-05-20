# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional, Any

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.model import load_tokenizer
from llamafactory.train.sft.metric import (
    ComputeAccuracy,
    ComputeClassification,
    ComputeSimilarity,
    eval_logit_processor,
)
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.callbacks import LogCallback, ReporterCallback


from llamafactory.hparams import get_train_args, read_args

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


from utils import get_task_combinations, create_vector_combination, plot_acc_coef_csv

from transformers import AutoModelForCausalLM

import gc
import numpy as np
import os
import pandas as pd
import time
import torch

TASKS = ["mnli", "qnli", "qqp", "sst2", "record"]
MODELS = ["llama-3.2-1b-instruct"]
METHODS = ["freeze"]
SEEDS = [42]
N_EVAL_POINTS = 41
EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 3
COEF_MAX = 2.0
DEVICE = "cuda"


logger = get_logger(__name__)


@torch.no_grad()
def apply_coef_inplace(model, base_state, tv, coef):
    """Set the merged weights to  base + coef * task_vector  *in place*.

    This replaces the old `TaskVector.apply_to`, which rebuilt a full fp32 model on
    every coefficient and never freed it (the CUDA caching allocator then fragmented
    and grew across the sweep). Here a single model is loaded once and only the merged
    keys are overwritten. `base_state` / `tv` are fp32 CPU tensors; `p.copy_` handles
    the host->device transfer and the cast to the model's (bf16) dtype.
    """
    for name, p in model.named_parameters():
        if name in tv:
            p.copy_(base_state[name] + coef * tv[name])


def build_trainer(model, template, dataset_module, tokenizer_module, data_args, model_args,
                  training_args, finetuning_args, generating_args, callbacks):
    """Build a Trainer + gen_kwargs for one eval dataset.

    Created ONCE per task and reused across all scaling coefficients (the model's
    weights are mutated in place). Constructing a fresh Trainer/Accelerator per
    coefficient was the second leak: each one registers forward hooks and pushes state
    into accelerate's global singletons that are never torn down.
    """
    tokenizer = tokenizer_module["tokenizer"]

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)

        if finetuning_args.compute_classification_metrics:
            metric_module["compute_metrics"] = ComputeClassification(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    return trainer, gen_kwargs


def predict_accuracy(trainer, dataset_module, gen_kwargs):
    predict_results = trainer.predict(
        dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs
    )
    return predict_results.metrics["predict_accuracy"]


def run_eval(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    task_combinations = get_task_combinations(TASKS)
    scaling_coef_range = np.linspace(0.0, COEF_MAX, N_EVAL_POINTS)[1:]
    print(scaling_coef_range)

    print("Task combinations:", task_combinations)

    task_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    selected_combination = list(task_combinations[task_idx])
    print(f"Running task combination index {task_idx}: {selected_combination}")

    compute_dtype = getattr(model_args, "compute_dtype", None) or torch.bfloat16

    for model in MODELS:
        for methods in METHODS:
            for seed in SEEDS:
                for tasks in [selected_combination]:
                    save_dir = f"saves_bts_merged/{methods}/{model}/{'_'.join(tasks)}_{seed}_best"
                    training_args.output_dir = save_dir
                    os.makedirs(save_dir, exist_ok=True)

                    template = get_template_and_fix_tokenizer(tokenizer, data_args)

                    print("Loading datasets for tasks:", tasks)
                    dataset_modules = {}
                    for task in tasks:
                        data_args.eval_dataset = [f"{task}_eval"]
                        dataset_modules[task] = get_dataset(
                            template, model_args, data_args, training_args, stage="sft", **tokenizer_module
                        )

                    print(
                        f"Creating task vector for model: {model}, method: {methods}, seed: {seed}, tasks: {tasks}"
                    )
                    (task_comb, task_vector), = create_vector_combination(model, methods, seed, tasks).items()
                    print("*" * 10 + str(task_comb) + " created" + "*" * 10)

                    # --- Load the base model ONCE; cache fp32 base weights for merged keys ---
                    pretrained_ckpt = f"saves_pretrained_weights/{methods}/{model}/pretrained_weights_{seed}"
                    tv = task_vector.vector
                    base_model = AutoModelForCausalLM.from_pretrained(
                        pretrained_ckpt, torch_dtype=torch.float32
                    )
                    base_state = {
                        k: v.detach().clone()
                        for k, v in base_model.state_dict().items()
                        if k in tv
                    }
                    merged_model = base_model.to(dtype=compute_dtype, device=DEVICE)
                    merged_model.eval()

                    if training_args.predict_with_generate:
                        tokenizer.padding_side = "left"  # use left-padding in generation

                    # --- Build one Trainer per task, reused across all coefficients ---
                    trainers = {
                        task: build_trainer(
                            merged_model, template, dataset_modules[task], tokenizer_module,
                            data_args, model_args, training_args, finetuning_args,
                            generating_args, callbacks,
                        )
                        for task in tasks
                    }

                    acc_coef = {}
                    best_avg_acc = 0.0
                    not_best_counter = 0
                    coef_selection_start = time.perf_counter()
                    for scaling_coef in scaling_coef_range:
                        print(f"Evaluating at scaling coefficient: {scaling_coef:.2f}")
                        apply_coef_inplace(merged_model, base_state, tv, scaling_coef)

                        acc_dict = {}
                        for task in tasks:
                            print(f"Evaluating on task: {task} with scaling coefficient: {scaling_coef:.2f}")
                            trainer, gen_kwargs = trainers[task]
                            acc_dict[task] = predict_accuracy(trainer, dataset_modules[task], gen_kwargs)

                        avg_acc = sum(acc_dict.values()) / len(acc_dict)
                        acc_coef.setdefault(scaling_coef, {}).update(acc_dict)

                        if avg_acc > best_avg_acc:
                            best_avg_acc = avg_acc
                            not_best_counter = 0
                            merged_model.save_pretrained(save_dir)
                            print("*" * 10 + f" New best average accuracy: {best_avg_acc:.4f} at scaling coefficient {scaling_coef:.2f}. Model saved. " + "*" * 10)
                        else:
                            not_best_counter += 1

                        print(f"Average accuracy at scaling coefficient {scaling_coef:.2f}: {avg_acc:.4f}")

                        gc.collect()
                        torch.cuda.empty_cache()

                        if EARLY_STOPPING and not_best_counter >= EARLY_STOPPING_PATIENCE:
                            print(f"Early stopping at scaling coefficient {scaling_coef:.2f} due to no improvement in average accuracy for {EARLY_STOPPING_PATIENCE} consecutive evaluations.")
                            break

                    coef_selection_elapsed = time.perf_counter() - coef_selection_start
                    print("*" * 10 + f" Best average accuracy for task combination {task_comb}: {best_avg_acc:.4f} " + "*" * 10)
                    print(f"Coefficient selection took {coef_selection_elapsed:.2f}s ({coef_selection_elapsed/60:.2f} min) for task combination {task_comb}")

                    acc_coef_df = pd.DataFrame.from_dict(acc_coef, orient="index").sort_index()
                    acc_coef_df.index.name = "scaling_coef"

                    csv_path = f"{save_dir}/{'_'.join(tasks)}_{seed}_acc_coef.csv"
                    acc_coef_df.to_csv(csv_path)
                    plot_acc_coef_csv(csv_path)

                    # --- Release everything before the next task combination ---
                    for trainer, _ in trainers.values():
                        trainer.accelerator.free_memory()
                    del trainers, merged_model, base_model, base_state, tv, task_vector
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = read_args()
    callbacks: list[Any] = []
    model_args, data_args, training_args, finetuning_args, generating_args, _ = get_train_args(args)

    callbacks.append(LogCallback())
    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last
    training_args.predict_with_generate = True
    training_args.do_predict = True

    run_eval(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
