from transformers import AutoModelForCausalLM
from task_vector import TaskVector

import torch


def create_task_vector(
    pretrained_checkpoint: str, finetuned_checkpoint: str
) -> TaskVector:
    task_vector = TaskVector(
        pretrained_checkpoint=pretrained_checkpoint,
        finetuned_checkpoint=finetuned_checkpoint,
        target_modules=["lora_A", "lora_B"],
    )

    return task_vector


task_vector = create_task_vector(
    "saves_pretrained_weights/lora/llama-3.2-1b-instruct/pretrained_weights_42",
    "saves_bts_preliminary/lora/llama-3.2-1b-instruct/train_qnli_42_1773148413",
)
model = task_vector.apply_to(
    "saves_pretrained_weights/lora/llama-3.2-1b-instruct/pretrained_weights_42",
    scaling_coef=1.0,
    args={"device": "cpu"},
)
# print(task_vector.vector["base_model.model.model.layers.9.self_attn.v_proj.lora_A.weight"])
model.save_pretrained("test")


model1 = AutoModelForCausalLM.from_pretrained(
    "saves_bts_preliminary/lora/llama-3.2-1b-instruct/train_qnli_42_1773148413"
)
model2 = AutoModelForCausalLM.from_pretrained("test")
model3 = AutoModelForCausalLM.from_pretrained(
    "saves_pretrained_weights/lora/llama-3.2-1b-instruct/pretrained_weights_42"
)

# print(model1.state_dict().keys())
# print(model2.state_dict().keys())
# print(model3.state_dict().keys())


print(model1.state_dict()["model.layers.15.self_attn.k_proj.lora_B.default.weight"])
print(model2.state_dict()["model.layers.15.self_attn.k_proj.lora_B.default.weight"])
print(model3.state_dict()["model.layers.15.self_attn.k_proj.lora_B.default.weight"])

print((model1.state_dict()["model.layers.15.self_attn.k_proj.lora_B.default.weight"] - model2.state_dict()["model.layers.15.self_attn.k_proj.lora_B.default.weight"]).mean())