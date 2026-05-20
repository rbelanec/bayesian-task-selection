from transformers import AutoModelForCausalLM
from task_vector import TaskVector

import torch


def create_task_vector(
    pretrained_checkpoint: str, finetuned_checkpoint: str
) -> TaskVector:
    task_vector = TaskVector(
        pretrained_checkpoint=pretrained_checkpoint,
        finetuned_checkpoint=finetuned_checkpoint,
        target_modules=["self_attn", "mlp"],
    )

    return task_vector


task_vector = create_task_vector(
    "saves_pretrained_weights/base/llama-3.2-1b-instruct/pretrained_weights_42",
    "saves_bts_preliminary/base/llama-3.2-1b-instruct/train_qnli_42_1776331409",
)
model = task_vector.apply_to(
    "saves_pretrained_weights/base/llama-3.2-1b-instruct/pretrained_weights_42",
    scaling_coef=1.0,
    args={"device": "cpu"},
)
# print(task_vector.vector["base_model.model.model.layers.9.self_attn.v_proj.lora_A.weight"])
model.save_state("test_base")


model1 = AutoModelForCausalLM.from_pretrained(
    "saves_bts_preliminary/base/llama-3.2-1b-instruct/train_qnli_42_1776331409"
)
model2 = AutoModelForCausalLM.from_pretrained("test_base")
model3 = AutoModelForCausalLM.from_pretrained(
    "saves_pretrained_weights/base/llama-3.2-1b-instruct/pretrained_weights_42"
)

# the_model = model3

# for key in the_model.state_dict():
#     if the_model.state_dict()[key].dtype == torch.int64:
#         print(the_model.state_dict()[key])

#     elif the_model.state_dict()[key].dtype == torch.uint8:
#         print(the_model.state_dict()[key])

#     else:
#         print(the_model.state_dict()[key].dtype)
    

print("done")

# print(model1.state_dict().keys())
print(model2.state_dict().keys())
# print(model3.state_dict().keys())


print(model1.state_dict()["model.layers.15.self_attn.k_proj.weight"])
print(model2.state_dict()["model.layers.15.self_attn.k_proj.weight"])
print(model3.state_dict()["model.layers.15.self_attn.k_proj.weight"])

print((model1.state_dict()["model.layers.15.self_attn.k_proj.weight"] - model2.state_dict()["model.layers.15.self_attn.k_proj.weight"]).mean())
