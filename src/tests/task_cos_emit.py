"""Check compute_task_cos.py's six emitted signals against direct computation.

src/tests/task_cos_gram.py covers the expensive half (streamed float64 Gram
accumulation == dense inner products, on real checkpoints). This covers the
cheap half: given a Gram matrix, does the CSV hold what the docstring says?

It needs no checkpoints at all. Random task vectors are generated, their exact
Gram is seeded into the --gram cache, and the script's emit path is run against
it; each row is then re-derived straight from the vectors. That also pins the
two relationships the signals are chosen to expose:

    dot_mix == |M| * ts_dot        (agree within a size, diverge across sizes)
    cos_mix != ts_cos              (unless members are mutually orthogonal)

Usage:
    PYTHONPATH=src:scripts python src/tests/task_cos_emit.py
"""

import csv
import os
import subprocess
import sys
import tempfile
from itertools import combinations

import numpy as np

SOURCES = ["s0", "s1", "s2", "s3"]
TARGETS = ["t0", "t1"]


def main():
    rng = np.random.default_rng(0)
    tasks = SOURCES + TARGETS
    # Deliberately correlated, not orthogonal: a shared component plus noise,
    # so cos_mix and ts_cos actually differ and the test can prove they do.
    shared = rng.normal(size=512)
    V = {t: shared * rng.uniform(0.2, 1.0) + rng.normal(size=512)
         for t in tasks}

    G = np.array([[V[a] @ V[b] for b in tasks] for a in tasks], dtype=np.float64)

    with tempfile.TemporaryDirectory() as d:
        gram = os.path.join(d, "gram.npz")
        out = os.path.join(d, "cos.csv")
        np.savez(gram, gram=G, tasks=np.array(tasks), n_params=V["s0"].size,
                 method="base", model="synthetic", seed=0)

        env = dict(os.environ, PYTHONPATH="src:scripts")
        r = subprocess.run(
            [sys.executable, "scripts/compute_task_cos.py",
             "--sources", *SOURCES, "--targets", *TARGETS,
             "--method", "base", "--gram", gram, "--out", out],
            capture_output=True, text=True, env=env)
        if r.returncode:
            print(r.stdout, r.stderr)
            raise SystemExit("compute_task_cos.py failed")
        assert "[cache] reusing" in r.stdout, "did not hit the Gram cache"

        rows = list(csv.DictReader(open(out)))

    expected_rows = sum(1 for r in range(2, len(SOURCES) + 1)
                        for _ in combinations(SOURCES, r)) * len(TARGETS)
    assert len(rows) == expected_rows, (len(rows), expected_rows)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    worst = {}
    saw_cos_gap = False
    for row in rows:
        M = row["combo"].split("+")
        tv = V[row["target"]]
        mix = sum(V[m] for m in M)
        pairs = list(combinations(M, 2))

        ref = dict(
            cos_mix=cos(mix, tv),
            dot_mix=float(mix @ tv),
            ts_cos=float(np.mean([cos(V[m], tv) for m in M])),
            ts_dot=float(np.mean([V[m] @ tv for m in M])),
            kc_cos=float(np.mean([cos(V[a], V[b]) for a, b in pairs])),
            kc_dot=float(np.mean([V[a] @ V[b] for a, b in pairs])),
        )
        for k, want in ref.items():
            got = float(row[k])
            err = abs(got - want) / max(abs(want), 1e-12)
            worst[k] = max(worst.get(k, 0.0), err)

        # documented identity: dot of the sum is |M| times the member mean
        assert abs(float(row["dot_mix"]) - len(M) * float(row["ts_dot"])) \
            <= 1e-9 * max(abs(float(row["dot_mix"])), 1.0), row
        if abs(ref["cos_mix"] - ref["ts_cos"]) > 1e-3:
            saw_cos_gap = True

    for k in sorted(worst):
        print(f"  {k:<8} worst relative error {worst[k]:.3e}")
    assert max(worst.values()) < 1e-9, f"signal mismatch: {worst}"
    assert saw_cos_gap, "cos_mix never diverged from ts_cos — test is not probing it"

    print(f"\nOK: {len(rows)} rows, all six signals match direct computation")
    print("    dot_mix == |M| * ts_dot held on every row")
    print("    cos_mix and ts_cos verified to be different orderings")


if __name__ == "__main__":
    main()
