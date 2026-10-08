"""Extract DTED training-log key metrics at sampled steps.

Reads the (potentially very large) train.log, dedupes per-step lines,
and prints selected metrics at sample steps to stdout / /tmp/curve.txt.
"""

from __future__ import annotations

import re

LOG = "/dockerdata/specforge_outputs/qwen3-4b-dted-b16init-1ep-lr1e5/train.log"
OUT = "/tmp/curve.txt"

step_pat = re.compile(r"step (\d+): (\{[^}]+\})")
seen: dict[int, str] = {}
with open(LOG, "rb") as f:
    buf = b""
    while True:
        chunk = f.read(64 * 1024 * 1024)
        if not chunk:
            break
        buf += chunk
        last_nl = buf.rfind(b"\n")
        if last_nl < 0:
            continue
        text = buf[: last_nl + 1].decode("utf-8", errors="ignore")
        buf = buf[last_nl + 1 :]
        for m in step_pat.finditer(text):
            step = int(m.group(1))
            if step not in seen:
                seen[step] = m.group(2)
    if buf:
        text = buf.decode("utf-8", errors="ignore")
        for m in step_pat.finditer(text):
            step = int(m.group(1))
            if step not in seen:
                seen[step] = m.group(2)

steps = sorted(seen)
print(f"total unique steps: {len(steps)}, min={steps[0]}, max={steps[-1]}")

sample_targets = [10, 100, 500, 1000, 2000, 5000, 10000, 15000,
                  20000, 25000, 30000, 35000, 40000, 45000, steps[-1]]
sample_steps = []
for tgt in sample_targets:
    for s in steps:
        if s >= tgt:
            if s not in sample_steps:
                sample_steps.append(s)
            break

rows = [
    f"{'step':>7} {'loss':>8} {'expAL':>7} {'gap':>6} {'p_tgt':>7} "
    f"{'P_mean':>7} {'exit_w':>7} {'acc':>7} {'lr':>10} {'grad':>7}"
]
for s in sample_steps:
    try:
        d = eval(seen[s])
    except Exception as exc:
        rows.append(f"{s:>7} PARSE ERROR: {exc}")
        continue
    rows.append(
        f"{s:>7} "
        f"{d.get('train/loss', 0):>8.4f} "
        f"{d.get('train/expected_al_mean', 0):>7.3f} "
        f"{d.get('train/gap_depth_mean', 0):>6.3f} "
        f"{d.get('train/p_tgt_mean', 0):>7.4f} "
        f"{d.get('train/P_mean', 0):>7.4f} "
        f"{d.get('train/exit_weight_sum_mean', 0):>7.4f} "
        f"{d.get('train/acc', 0):>7.4f} "
        f"{d.get('train/lr', 0):>10.2e} "
        f"{d.get('train/grad_norm', 0):>7.4f}"
    )

with open(OUT, "w") as f:
    for r in rows:
        f.write(r + "\n")

for r in rows:
    print(r)
