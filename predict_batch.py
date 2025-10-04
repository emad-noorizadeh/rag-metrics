# scripts/predict_batch.py
"""
Load & score a processed set

python scripts/predict_batch.py \
  --model artifacts/lr_model.pkl \
  --npz processed/test_v1.npz \
  --out artifacts/test_scores.jsonl
"""
import argparse, json, numpy as np
from classifier import RagFaithfulnessClassifier
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Pickle saved by eval_kfold.py")
    ap.add_argument("--npz", required=True, help="Processed NPZ (X,y,feature_names,q_list,a_list,idx)")
    ap.add_argument("--out", default=None, help="Optional JSON lines of scores")
    args = ap.parse_args()

    # Load model
    clf = RagFaithfulnessClassifier.load_from_pickle(args.model)

    # If you want to score *raw text*, use your data_processing pipeline instead.
    # Here we score the NPZ directly to keep it fast and deterministic.
    npz = np.load(args.npz, allow_pickle=True)
    X = npz["X"]
    y = npz.get("y", None)
    q_list = list(npz.get("q_list", []))
    a_list = list(npz.get("a_list", []))
    idxs   = list(npz.get("idx", []))

    # Make sure feature order matches the model
    # (The artifact’s training order lives in clf.feature_names_.
    #  Your NPZ was saved with the same order by data_processing, so this should align.)
    # If you ever need re-projection, add it similarly to the class’ _reproject().

    probs = clf.model.predict_proba(X)[:, 1]
    thr = getattr(clf, "_inference_threshold", 0.5)
    preds = (probs >= thr).astype(int)

    # Print a quick summary
    print(f"Scored {len(probs)} rows. Using threshold={thr:.3f}.")
    if y is not None and len(y) == len(probs):
        acc = (preds == y).mean()
        print(f"  Accuracy: {acc:.3f}")

    # Optional write-out
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for i in range(len(probs)):
                row = {
                    "idx": int(idxs[i]) if i < len(idxs) else i,
                    "p": float(probs[i]),
                    "yhat": int(preds[i]),
                }
                if y is not None and i < len(y):
                    row["y"] = int(y[i])
                if i < len(q_list):
                    row["q"] = q_list[i]
                if i < len(a_list):
                    row["a"] = a_list[i]
                f.write(json.dumps(row) + "\n")
        print(f"Wrote scores → {args.out}")

if __name__ == "__main__":
    main()