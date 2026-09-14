# Adopted from https://github.com/danielm1405/iso-merging/blob/main/src/models/task_vectors.py

import torch

from transformers import AutoModelForCausalLM


def symmetric_difference(A: list, B: list):
    """Returns the symmetric difference between two lists."""
    return list(set(A) ^ set(B))


class TaskVector:
    def __init__(
        self,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        vector=None,
        target_modules=None,
    ):
        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )

            with torch.no_grad():
                # load pretrained weights
                pretrained_state_dict = AutoModelForCausalLM.from_pretrained(
                    pretrained_checkpoint
                ).state_dict()

                # load finetuned weights
                finetuned_state_dict = AutoModelForCausalLM.from_pretrained(
                    finetuned_checkpoint
                ).state_dict()

                # print(f"Pretrained checkpoint keys: {pretrained_state_dict.keys()}")
                # print(f"Finetuned checkpoint keys: {finetuned_state_dict.keys()}")

            assert pretrained_state_dict.keys() == finetuned_state_dict.keys(), (
                f"Pretrained and finetuned checkpoints have different keys: {symmetric_difference(pretrained_state_dict.keys(), finetuned_state_dict.keys())}"
            )

            self.vector = {}
            for key in pretrained_state_dict:
                if target_modules is not None and not any(
                    target_module in key for target_module in target_modules
                ):
                    continue

                self.vector[key] = (
                    finetuned_state_dict[key] - pretrained_state_dict[key]
                )

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return self.__class__(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def _load_checkpoint(self, checkpoint_path):
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.float32
        )
        return model

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, args=None):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = self._load_checkpoint(pretrained_checkpoint)
            device = args["device"] if isinstance(args, dict) else args.device
            pretrained_model = pretrained_model.to(device)

            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                pretrained_tensor = pretrained_state_dict[key]
                if key not in self.vector:
                    new_state_dict[key] = pretrained_tensor
                else:
                    tv_tensor = self.vector[key].to(
                        device=pretrained_tensor.device, dtype=torch.float32
                    )
                    new_state_dict[key] = (
                        pretrained_tensor.to(torch.float32) + scaling_coef * tv_tensor
                    )
        pretrained_model.load_state_dict(new_state_dict)
        return pretrained_model
