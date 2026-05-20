from task_vector import TaskVector

from iso import iso_c, iso_cts

import glob as glob_module
from itertools import combinations
from typing import Optional
from utils import get_task_combinations, create_vector_combinations

from transformers import AutoModelForCausalLM


tasks = ["mnli", "qnli", "qqp", "sst2", "record"]
models = ["llama-3.2-1b-instruct"]
# methods = ["lora"]
methods = ["base"]
seeds = [42]


task_combinations = get_task_combinations(tasks)


if __name__ == "__main__":
    for model in models:
        for methods in methods:
            for seed in seeds:
                task_vectors: dict[str, TaskVector] = {}
                for tasks in task_combinations:
                    print(
                        f"Creating task combination for model: {model}, method: {methods}, seed: {seed}, tasks: {tasks}"
                    )
                    task_vectors.update(
                        create_vector_combinations(model, methods, seed, tasks)
                    )

                    print("_".join(tasks))
                    merged_model = task_vectors["_".join(tasks)].apply_to(
                        f"saves_pretrained_weights/{methods}/{model}/pretrained_weights_{seed}",
                        scaling_coef=1.0,
                        args={"device": "cpu"},
                    )

                    save_dir = (
                        f"saves_bts_merged/{methods}/{model}/{'_'.join(tasks)}_{seed}"
                    )
                    merged_model.save_pretrained(save_dir)
