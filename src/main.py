from task_vector import TaskVector

from iso import iso_c, iso_cts

import glob as glob_module
from itertools import combinations

tasks = ["mnli", "qnli", "qqp", "sst2", "record"]
models = ["llama-3.2-1b-instruct"]
methods = ["lora"]
seeds = [42]

task_combinations = [
    c
    for r in range(2, len(tasks) + 1)
    for c in combinations(tasks, r)
]

print(task_combinations)

def create_task_vector(pretrained_checkpoint: str, finetuned_checkpoint: str) -> TaskVector:
    task_vector = TaskVector(
        pretrained_checkpoint=pretrained_checkpoint,
        finetuned_checkpoint=finetuned_checkpoint,
        target_modules=["lora_A", "lora_B"],
    )

    return task_vector

def create_task_combination(model, method, seed, tasks):
    pretrained_checkpoint = f"saves_pretrained_weights/{method}/{model}/pretrained_weights_{seed}"

    task_vectors = []
    for task in tasks:
        pattern = f"saves_bts_preliminary/{method}/{model}/train_{task}_{seed}_*"
        matches = glob_module.glob(pattern)
        
        if not matches:
            raise FileNotFoundError(f"No checkpoint found for task '{task}' matching: {pattern}")
        
        finetuned_checkpoint = matches[0]
        task_vectors.append(create_task_vector(pretrained_checkpoint, finetuned_checkpoint))
    
    return {"_".join(tasks): sum(task_vectors)}


if __name__ == "__main__":
    for model in models:
        for methods in methods:
            for seed in seeds:

                task_vectors = {}
                for tasks in task_combinations:
                    print(f"Creating task combination for model: {model}, method: {methods}, seed: {seed}, tasks: {tasks}")
                    task_vectors.update(create_task_combination(model, methods, seed, tasks))

                    print("_".join(tasks))
                    
                    # print(task_vectors["_".join(tasks)].vector)


                merged_model = task_vectors["mnli_qnli"].apply_to("saves_pretrained_weights/lora/llama-3.2-1b-instruct/pretrained_weights_42", scaling_coef=1.0, args={"device": "cpu"})

