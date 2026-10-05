"""Accuracy harness: runs the engine on N labelled synthetic frames and reports P/R/F1/FP/FN."""
import argparse, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "simulator")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vision-engine"))
from generate_test_streams import SCEN, build
from engine import Engine

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--detector", default="color"); ap.add_argument("--weights", default="yolov8n.pt")
a = ap.parse_args(); eng, rng = Engine(a.detector, a.weights), random.Random(a.seed)
tp = fp = fn = tn = 0; per = {}
for i in range(a.n):
    scn = rng.choice(list(SCEN)); frame, meta = build(scn, rng.randint(0, 10**6))
    pred = eng.analyze(frame, meta)[0]["alert"]; truth = meta["truth_alert"]
    tp += pred and truth; fp += pred and not truth; fn += (not pred) and truth; tn += (not pred) and not truth
    per.setdefault(scn, [0, 0]); per[scn][0] += pred == truth; per[scn][1] += 1
p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
print(f"N={a.n}  TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"Precision={p:.3f} Recall={r:.3f} F1={2*p*r/(p+r) if p+r else 0:.3f}  FPR={fp/(fp+tn) if fp+tn else 0:.3f} FNR={fn/(fn+tp) if fn+tp else 0:.3f}")
for k, (c, t) in per.items(): print(f"  {k:14s} accuracy {c}/{t}")
