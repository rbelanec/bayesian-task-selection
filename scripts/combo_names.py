# Copyright 2025 the Anonymous Authors.
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

"""Turn a merge-sweep directory/CSV stem back into the task names it encodes.

Save dirs are named ``"_".join(tasks)``, which stops being invertible the moment
a task name contains an underscore: ``"anli_r1_hellaswag".split("_")`` yields
three tokens for two tasks. That was safe for the original five sources
(mnli/qnli/qqp/sst2/record) and became silently wrong when anli_r1 and squad_v2
joined them — inflating ``n_tasks`` for every combo containing either, and
breaking the task-vector lookup in compute_target_sar.py outright.

Parsing lives here rather than in each consumer so summarize_target.py and
compute_target_sar.py cannot drift apart, the same reasoning eval_target.py
gives for importing eval.py instead of copying its helpers.
"""

from __future__ import annotations

import ast
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_TARGET = os.path.join(REPO_ROOT, "src", "eval", "eval_target.py")

_task_lists: dict[tuple[str, str], list[str]] = {}


def eval_target_literal(name, path=EVAL_TARGET):
    """Value of a top-level ``NAME = <literal>`` assignment, read without importing.

    eval_target.py pulls in torch and llamafactory at import time; the analysis
    scripts deliberately stay dependency-free, so the value is read via AST —
    the same trick scripts/combo_status.py uses to mirror the sweep's indexing.
    Serves the task lists below and the sweep constants (COEF_MAX,
    N_EVAL_POINTS) that scripts/coef_distribution.py needs.
    """
    key = (path, name)
    if key not in _task_lists:
        tree = ast.parse(open(path).read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                _task_lists[key] = ast.literal_eval(node.value)
                break
        else:
            raise SystemExit(f"could not read {name} from {path}")
    return _task_lists[key]


def source_tasks():
    """SOURCE_TASKS as declared in src/eval/eval_target.py."""
    return list(eval_target_literal("SOURCE_TASKS"))


def target_tasks():
    """TARGET_TASKS as declared in src/eval/eval_target.py."""
    return list(eval_target_literal("TARGET_TASKS"))


def coef_grid():
    """The coefficient grid the sweep evaluates: (values, step).

    Mirrors ``np.linspace(0, COEF_MAX, N_EVAL_POINTS)[1:]`` in eval.py /
    eval_target.py — the leading 0.0 is dropped there because a zero coefficient
    is just the pretrained model.
    """
    coef_max = eval_target_literal("COEF_MAX")
    n_points = eval_target_literal("N_EVAL_POINTS")
    step = coef_max / (n_points - 1)
    return [step * i for i in range(1, n_points)], step


def parse_combo(stem, tasks=None):
    """Split a ``"_".join(tasks)`` stem back into its task names.

    Longest name first, so ``squad_v2`` is matched ahead of a bare ``squad``
    prefix. Greedy matching is exact for the current vocabulary because no task
    name is a prefix of another; a vocabulary where that stopped holding would
    need backtracking. An unparseable stem raises rather than degrading to a
    plausible-looking split — a silent mis-split is what put a wrong ``n_tasks``
    on half the grid.
    """
    vocab = sorted(source_tasks() if tasks is None else tasks, key=len, reverse=True)
    out, rest = [], stem
    while rest:
        for t in vocab:
            if rest == t or rest.startswith(t + "_"):
                out.append(t)
                rest = rest[len(t) + 1:]
                break
        else:
            raise ValueError(f"unknown task in combo {stem!r} (stuck at {rest!r})")
    return out


# Canonical display names for the paper's figures and tables. Kept here, beside the
# task lists, so a figure axis and a LaTeX header can never disagree about what a
# dataset is called.
DISPLAY_NAMES = {
    "mnli": "MNLI", "qnli": "QNLI", "qqp": "QQP", "sst2": "SST-2",
    "record": "ReCoRD", "mrpc": "MRPC", "boolq": "BoolQ", "rte": "RTE",
    "cola": "CoLA", "snli": "SNLI", "anli_r1": "ANLI-R1", "paws": "PAWS",
    "imdb": "IMDb", "squad_v2": "SQuAD v2", "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande", "cb": "CB", "scitail": "SciTail",
    "stsb": "STS-B", "cr": "CR", "rotten_tomatoes": "Rotten Tomatoes",
    "multirc": "MultiRC", "copa": "COPA", "piqa": "PIQA",
}


def display_name(task: str) -> str:
    """Pretty name for a task id, or the id itself if it has no entry."""
    return DISPLAY_NAMES.get(task, task)


def display_combo(combo: str, sep: str = " + ") -> str:
    """"mnli+anli_r1" -> "MNLI + ANLI-R1"."""
    return sep.join(display_name(t) for t in combo.split("+"))
