# Adopted from https://github.com/danielm1405/iso-merging/blob/main/src/models/task_vectors.py

import torch
from safetensors.torch import load_file

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
                pretrained_state_dict = self._safe_load(f"{pretrained_checkpoint}/adapter_model.safetensors")

                # load finetuned weights
                finetuned_state_dict = self._safe_load(f"{finetuned_checkpoint}/adapter_model.safetensors")

            assert pretrained_state_dict.keys() == finetuned_state_dict.keys(), (
                f"Pretrained and finetuned checkpoints have different keys: {symmetric_difference(pretrained_state_dict.keys(), finetuned_state_dict.keys())}"
            )

            self.vector = {}
            for key in pretrained_state_dict:
                if pretrained_state_dict[key].dtype == torch.int64:
                    continue
                if pretrained_state_dict[key].dtype == torch.uint8:
                    continue

                if target_modules is not None and not any(
                    target_module in key for target_module in target_modules
                ):
                    continue

                self.vector[key] = (
                    finetuned_state_dict[key] - pretrained_state_dict[key]
                )

    def _safe_load(self, checkpoint_path):
        try:
            return load_file(checkpoint_path, device="cpu")
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            raise

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

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return self.__class__(vector=new_vector)

    def __pow__(self, power):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] ** power
        return self.__class__(vector=new_vector)

    def __mul__(self, other):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = other * self.vector[key]
        return self.__class__(vector=new_vector)

    def dot(self, other):
        """Dot product of two task vectors."""
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))
    
    def _load_checkpoint(self, checkpoint_path):
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
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
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"
                    )
                    new_state_dict[key] = pretrained_state_dict[key].to(device)
                else:
                    new_state_dict[key] = (
                        pretrained_state_dict[key].to(self.vector[key].device) + scaling_coef * self.vector[key]
                    )
        pretrained_model.load_state_dict(new_state_dict)
        return pretrained_model