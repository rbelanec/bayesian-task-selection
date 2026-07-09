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

"""Evaluate each SOURCE-task mixture on the held-out TARGET tasks.

Companion to eval.py: that script sweeps the merge scaling coefficient and
measures the merged model on its own (source) tasks. This one rebuilds the same
summed task vector per source combination and sweeps the same coefficient grid,
but evaluates on the TARGET tasks instead — how well does each source mixture
transfer? No model is saved (the best-on-source merged checkpoint already lives
in the sibling ``*_best`` dir); the sweep always runs the full grid so the
target curves are complete.

Output per combination:
    saves_bts_merged/{method}/{model}/{tasks}_{seed}_target/
        {tasks}_{seed}_target_acc_coef.csv   (+ .png)

with one column per target task. The scaling_coef index is identical to the
source sweep CSV in ``{tasks}_{seed}_best/``, so the two join on scaling_coef
(e.g. to read target accuracy at the source-selected best coefficient, or to
pick a per-target best coefficient).

Usage (one SLURM array index per source combination, same indexing as eval.py):
    sbatch --array=0-25 scripts/slurm/eval_target_array.sh
"""

from typing import TYPE_CHECKING, Optional, Any

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.logging import get_logger
from llamafactory.model import load_tokenizer
from llamafactory.train.callbacks import LogCallback, ReporterCallback

from llamafactory.hparams import get_train_args, read_args

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


# eval.py sits in this same directory, so when this file runs as a script
# (sys.path[0] = src/eval) `eval` resolves to it; the merge/eval helpers are
# shared rather than copied so the two sweeps cannot drift apart.
from eval import apply_coef_inplace, build_trainer, predict_accuracy
from utils import get_task_combinations, create_vector_combination, plot_acc_coef_csv

from transformers import AutoModelForCausalLM

import gc
import numpy as np
import os
import pandas as pd
import time
import torch

SOURCE_TASKS = ["mnli", "qnli", "qqp", "sst2", "record"]
TARGET_TASKS = ["mrpc", "boolq", "rte", "cola"]
MODELS = ["llama-3.2-1b-instruct"]
METHODS = ["base"]
SEEDS = [42]
N_EVAL_POINTS = 41
COEF_MAX = 2.0
DEVICE = "cuda"

MODEL_MAPPING = {
        "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct"
}


logger = get_logger(__name__)


def run_eval(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    task_combinations = get_task_combinations(SOURCE_TASKS)
    scaling_coef_range = np.linspace(0.0, COEF_MAX, N_EVAL_POINTS)[1:]
    print(scaling_coef_range)

    print("Source task combinations:", task_combinations)
    print("Target tasks:", TARGET_TASKS)

    task_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    selected_combination = list(task_combinations[task_idx])
    print(f"Running source combination index {task_idx}: {selected_combination}")

    for model in MODELS:
        model_args.model_name_or_path = MODEL_MAPPING[model]

        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        compute_dtype = getattr(model_args, "compute_dtype", None) or torch.bfloat16

        for methods in METHODS:
            for seed in SEEDS:
                for tasks in [selected_combination]:
                    save_dir = f"saves_bts_merged/{methods}/{model}/{'_'.join(tasks)}_{seed}_target"
                    training_args.output_dir = save_dir
                    os.makedirs(save_dir, exist_ok=True)

                    template = get_template_and_fix_tokenizer(tokenizer, data_args)

                    print("Loading datasets for target tasks:", TARGET_TASKS)
                    dataset_modules = {}
                    for task in TARGET_TASKS:
                        data_args.eval_dataset = [f"{task}_eval"]
                        dataset_modules[task] = get_dataset(
                            template, model_args, data_args, training_args, stage="sft", **tokenizer_module
                        )

                    print(
                        f"Creating task vector for model: {model}, method: {methods}, seed: {seed}, source tasks: {tasks}"
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

                    # --- Build one Trainer per target task, reused across all coefficients ---
                    trainers = {
                        task: build_trainer(
                            task, merged_model, template, dataset_modules[task], tokenizer_module,
                            data_args, model_args, training_args, finetuning_args,
                            generating_args, callbacks,
                        )
                        for task in TARGET_TASKS
                    }

                    acc_coef = {}
                    best_avg_acc = 0.0
                    sweep_start = time.perf_counter()
                    for scaling_coef in scaling_coef_range:
                        print(f"Evaluating at scaling coefficient: {scaling_coef:.2f}")
                        apply_coef_inplace(merged_model, base_state, tv, scaling_coef)

                        acc_dict = {}
                        for task in TARGET_TASKS:
                            print(f"Evaluating on target task: {task} with scaling coefficient: {scaling_coef:.2f}")
                            trainer, gen_kwargs = trainers[task]
                            acc_dict[task] = predict_accuracy(trainer, dataset_modules[task], gen_kwargs)
                            print(f"Accuracy for target task {task} at scaling coefficient {scaling_coef:.2f}: {acc_dict[task]:.4f}")

                        avg_acc = sum(acc_dict.values()) / len(acc_dict)
                        acc_coef.setdefault(scaling_coef, {}).update(acc_dict)
                        best_avg_acc = max(best_avg_acc, avg_acc)  # tracking only; no model is saved here

                        print(f"Average target accuracy at scaling coefficient {scaling_coef:.2f}: {avg_acc:.4f}")

                        gc.collect()
                        torch.cuda.empty_cache()

                    sweep_elapsed = time.perf_counter() - sweep_start
                    print("*" * 10 + f" Best average target accuracy for source combination {task_comb}: {best_avg_acc:.4f} " + "*" * 10)
                    print(f"Target sweep took {sweep_elapsed:.2f}s ({sweep_elapsed/60:.2f} min) for source combination {task_comb}")

                    acc_coef_df = pd.DataFrame.from_dict(acc_coef, orient="index").sort_index()
                    acc_coef_df.index.name = "scaling_coef"

                    csv_path = f"{save_dir}/{'_'.join(tasks)}_{seed}_target_acc_coef.csv"
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
