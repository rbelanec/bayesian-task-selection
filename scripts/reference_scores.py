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

"""The single-task reference score a merged accuracy is measured against.

Both summarize scripts need the zero-shot floor and single-task-fine-tune
ceiling for a task, and getting one wrong silently rescales every retention
ratio and NAI built on it. Two corrections are needed and neither is obvious,
so they live here once rather than in each script:

  * which key to read — exact_match for the classification tasks, pearsonr for
    the regression ones (stsb has no exact_match at all; its entry in
    scripts/compute_metrics.py is [pearsonr, spearmanr]). Same
    `exact_match.fillna(pearsonr)` convention compute_metrics.py uses for its
    own `performance` column.
  * which scale it is on — `evaluate`'s squad_v2 reports 0-100 while every other
    task is 0-1, so it needs a /100 to be comparable with a merged accuracy.
    compute_metrics.py applies the same correction when building results.tex.

Read `compute_metrics.jsonl`, never `predict_results.json`: the latter holds the
trainer's naive per-row exact-match, which is wrong for the tasks with a
task-specific metric (record 0.5097 vs the correct 0.7878; squad_v2 0.7356 vs
0.8258).

`src/utils.py` carries an equivalent `_eval_exact_match` for the torch-side code
path — kept separate because these scripts stay importable without torch. Keep
PERCENT_SCALE_TASKS in sync with it and with src/metrics.py.
"""

from __future__ import annotations

import glob
import json

# Tasks whose stored score is a 0-100 percentage rather than a 0-1 fraction.
# Mirrors src/utils.py:PERCENT_SCALE_TASKS.
PERCENT_SCALE_TASKS = frozenset({"squad_v2"})


def reference_score(pattern, task=None):
    """Headline score from the first compute_metrics.jsonl matching, else None.

    Lines are merged before reading (a task may log f1 on one line and
    exact_match on another). `task` selects the scale correction; pass it for
    anything that might be in PERCENT_SCALE_TASKS.
    """
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    metrics = {}
    with open(hits[0]) as f:
        for line in f:
            if line.strip():
                metrics.update(json.loads(line))

    score = metrics.get("exact_match")
    if score is None:
        score = metrics.get("pearsonr")
    if score is None:
        return None
    return score / 100 if task in PERCENT_SCALE_TASKS else score
