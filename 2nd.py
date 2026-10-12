"""
Backpropagation Neural Network Prediction and Feature Analysis System
Dataset : Breast Cancer Wisconsin (Diagnostic) - 569 samples, 30 features
Classes : Benign = 0, Malignant = 1
Usage   : python breast_cancer_bpnn.py [path_to_csv]
          Accepts the Kaggle data.csv (id, diagnosis, 30 features) or the UCI wdbc.data.
Needs   : numpy, pandas   (scikit-learn is used only as a last-resort fallback for the data)

Models (all use sigmoid activations, backpropagation and binary cross-entropy loss):
  BPNN  : 30-8-1       plain backprop, online learning (one sample per weight update)
  MLP   : 30-16-8-1    two hidden layers, mini-batch gradient descent
  SHLNN : 30-12-1      single hidden layer, mini-batch gradient descent
"""
import sys
import os
import time
import numpy as np
import pandas as pd

SEED = 42
TEST_FRACTION = 0.2
THRESHOLD = 0.5
LOW_DIST = 0.15      # |output - 0.5| <  0.15 -> Low confidence
HIGH_DIST = 0.35     # |output - 0.5| >= 0.35 -> High confidence (output < 0.15 or > 0.85)
PERTURB = 0.10       # feature perturbation (in normalized units) for contribution analysis

BASE = ["radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
FEATURES = [f"{b}_mean" for b in BASE] + [f"{b}_se" for b in BASE] + [f"{b}_worst" for b in BASE]
CLASS_NAMES = {0: "Benign", 1: "Malignant"}

CANDIDATE_FILES = ["data.csv", "breast_cancer.csv", "breast-cancer.csv", "wdbc.data",
                   "wdbc.csv", "breast_cancer_wisconsin.csv", "dataset.csv"]
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def parse_wdbc(path_or_url):
    """Return (X[569,30], y) or None if the file is not the breast-cancer dataset."""
    try:
        df = pd.read_csv(path_or_url, header=None, dtype=str)
    except Exception:
        return None
    label_col = None
    for c in df.columns:
        vals = set(df[c].dropna().astype(str).str.strip().unique())
        if {"M", "B"} <= vals and vals <= {"M", "B", "diagnosis"}:
            label_col = c
            break
    if label_col is None:
        return None
    df = df[df[label_col].isin(["M", "B"])]            # drops a header row if present
    feat = df.iloc[:, label_col + 1:].apply(pd.to_numeric, errors="coerce")
    feat = feat.dropna(axis=1, how="all")              # drops Kaggle's empty 'Unnamed: 32'
    if feat.shape[1] != 30 or feat.isna().any().any():
        return None
    y = (df[label_col].values == "M").astype(int)      # Malignant = 1, Benign = 0
    return feat.values.astype(float), y


def load_dataset(path=None):
    files = ([path] if path else []) + CANDIDATE_FILES
    for p in files:
        if p and os.path.exists(p):
            res = parse_wdbc(p)
            if res is not None:
                print(f"Loaded dataset from: {p}")
                return res
    res = parse_wdbc(UCI_URL)
    if res is not None:
        print("Downloaded dataset from UCI repository.")
        return res
    try:
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        print("Loaded dataset from scikit-learn (built in).")
        return d.data.astype(float), (1 - d.target).astype(int)   # sklearn: 0=malignant -> flip
    except Exception:
        pass
    print("ERROR: dataset not found. Put the Kaggle 'data.csv' next to this script or run:")
    print("       python breast_cancer_bpnn.py <path_to_csv>")
    sys.exit(1)


class Data:
    """Min-max normalised, stratified train/test split (scaler fitted on train only)."""
    def __init__(self, X, y):
        rng = np.random.default_rng(SEED)
        test_idx = []
        for c in (0, 1):
            idx = np.where(y == c)[0]
            k = int(round(len(idx) * TEST_FRACTION))
            test_idx += list(rng.choice(idx, size=k, replace=False))
        mask = np.zeros(len(y), dtype=bool)
        mask[test_idx] = True
        self.X_train_raw, self.y_train = X[~mask], y[~mask]
        self.X_test_raw, self.y_test = X[mask], y[mask]
        self.min = self.X_train_raw.min(axis=0)
        self.max = self.X_train_raw.max(axis=0)
        span = np.where(self.max - self.min == 0, 1.0, self.max - self.min)
        self.span = span
        self.X_train = (self.X_train_raw - self.min) / span
        self.X_test = (self.X_test_raw - self.min) / span
        self.train_mean = self.X_train.mean(axis=0)     # mean in normalised units


# ----------------------------------------------------------------------------
# Neural network with backpropagation
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class NeuralNet:
    def __init__(self, layers, lr=0.5, epochs=200, batch_size=16, seed=SEED, name="NN"):
        self.layers, self.lr, self.epochs = layers, lr, epochs
        self.batch_size, self.name = batch_size, name
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.W, self.b = [], []
        for n_in, n_out in zip(layers[:-1], layers[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))                   # Xavier initialisation
            self.W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))
        self.loss_history, self.acc_history = [], []
        self.train_time = 0.0

    def forward(self, X):
        """Return list of activations [input, hidden..., output]."""
        acts = [X]
        for W, b in zip(self.W, self.b):
            acts.append(sigmoid(acts[-1] @ W + b))
        return acts

    def predict_proba(self, X):
        return self.forward(X)[-1].ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= THRESHOLD).astype(int)

    def logit(self, X):
        """Pre-activation (weighted sum) of the output neuron; does not saturate like the sigmoid."""
        a = self.forward(X)[-2]
        return (a @ self.W[-1] + self.b[-1]).ravel()

    def hidden_activations(self, X):
        """Activations of the first hidden layer."""
        return self.forward(X)[1]

    @staticmethod
    def bce(p, y):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X, y):
        t0 = time.time()
        n = len(y)
        y = y.astype(float)
        for epoch in range(self.epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                xb, yb = X[idx], y[idx].reshape(-1, 1)
                acts = self.forward(xb)
                # sigmoid output + cross-entropy loss -> output error term = (a - y)
                delta = (acts[-1] - yb) / len(idx)
                for l in range(len(self.W) - 1, -1, -1):
                    gW = acts[l].T @ delta
                    gb = delta.sum(axis=0)
                    if l > 0:
                        delta = (delta @ self.W[l].T) * acts[l] * (1 - acts[l])
                    self.W[l] -= self.lr * gW
                    self.b[l] -= self.lr * gb
            p = self.predict_proba(X)
            self.loss_history.append(self.bce(p, y))
            self.acc_history.append(float(np.mean((p >= THRESHOLD) == y)) * 100)
        self.train_time = time.time() - t0
        return self

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y)) * 100


def build_model(kind):
    if kind == "BPNN":
        return NeuralNet([30, 8, 1], lr=0.1, epochs=60, batch_size=1, name="BPNN")
    if kind == "MLP":
        return NeuralNet([30, 16, 8, 1], lr=0.5, epochs=200, batch_size=16, name="MLP")
    if kind == "SHLNN":
        return NeuralNet([30, 12, 1], lr=0.5, epochs=200, batch_size=16, name="SHLNN")
    raise ValueError(kind)


class Trainer:
    """Trains each model lazily and caches it."""
    def __init__(self, data):
        self.data = data
        self.models = {}

    def get(self, kind):
        if kind not in self.models:
            print(f"[training {kind} ...]")
            self.models[kind] = build_model(kind).fit(self.data.X_train, self.data.y_train)
        return self.models[kind]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def confidence_category(p):
    d = abs(p - THRESHOLD)
    if d < LOW_DIST:
        return "Low Confidence"
    if d < HIGH_DIST:
        return "Moderate Confidence"
    return "High Confidence"


def metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / len(y_true) * 100
    prec = tp / (tp + fp) * 100 if tp + fp else 0.0
    rec = tp / (tp + fn) * 100 if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1)


def ask_int(prompt, default, lo, hi):
    try:
        v = int(input(prompt) or default)
    except ValueError:
        v = default
    return max(lo, min(hi, v))


# ----------------------------------------------------------------------------
# 1. Tumor classification using BPNN
# ----------------------------------------------------------------------------
def option1(tr):
    header("1. Tumor Classification using BPNN (30-8-1, sigmoid, backpropagation)")
    d = tr.data
    m = tr.get("BPNN")
    out = m.predict_proba(d.X_test)
    pred = (out >= THRESHOLD).astype(int)
    print("Training progress (epoch: loss / train accuracy):")
    for e in list(range(0, m.epochs, 10)) + [m.epochs - 1]:
        print(f"  epoch {e + 1:>3}: loss {m.loss_history[e]:.4f}   acc {m.acc_history[e]:.2f}%")
    n = ask_int(f"\nHow many test samples to display? (0-{len(pred)}, Enter = 20): ", 20, 0, len(pred))
    print(f"\n{'#':>4} {'Actual':<10} {'Predicted':<10} {'Output':>8}  Result")
    for i in range(n):
        ok = "OK" if pred[i] == d.y_test[i] else "WRONG"
        print(f"{i:>4} {CLASS_NAMES[d.y_test[i]]:<10} {CLASS_NAMES[pred[i]]:<10} {out[i]:>8.4f}  {ok}")
    mt = metrics(d.y_test, pred)
    print(f"\nTrain accuracy: {m.accuracy(d.X_train, d.y_train):.2f}%")
    print(f"Test accuracy : {mt['acc']:.2f}%  ({int(np.sum(pred != d.y_test))} misclassified of {len(pred)})")
    print(f"Confusion matrix  TN={mt['tn']}  FP={mt['fp']}  FN={mt['fn']}  TP={mt['tp']}")


# ----------------------------------------------------------------------------
# 2. Prediction confidence analyzer (MLP)
# ----------------------------------------------------------------------------
def option2(tr):
    header("2. Prediction Confidence Analyzer (MLP 30-16-8-1)")
    d = tr.data
    m = tr.get("MLP")
    out = m.predict_proba(d.X_test)
    pred = (out >= THRESHOLD).astype(int)
    cats = [confidence_category(p) for p in out]
    print(f"Confidence = distance of output from threshold {THRESHOLD}:")
    print(f"  Low < {LOW_DIST} | Moderate {LOW_DIST}-{HIGH_DIST} | High >= {HIGH_DIST}\n")
    print(f"{'Sample':>6} {'Output':>8} {'Predicted':<10} {'Distance':>9}  Confidence")
    for i in range(len(out)):
        print(f"{i:>6} {out[i]:>8.4f} {CLASS_NAMES[pred[i]]:<10} {abs(out[i] - THRESHOLD):>9.4f}  {cats[i]}")
    print("\nSummary:")
    for c in ["High Confidence", "Moderate Confidence", "Low Confidence"]:
        k = [i for i, x in enumerate(cats) if x == c]
        acc = np.mean(pred[k] == d.y_test[k]) * 100 if k else float("nan")
        print(f"  {c:<20}: {len(k):>3} samples   accuracy within group: {acc:.2f}%")
    print(f"\nMLP test accuracy: {m.accuracy(d.X_test, d.y_test):.2f}%")


# ----------------------------------------------------------------------------
# 3. Feature contribution analyzer
# ----------------------------------------------------------------------------
def option3(tr):
    header("3. Feature Contribution Analyzer (BPNN)")
    d = tr.data
    m = tr.get("BPNN")
    i = ask_int(f"Test sample number (0-{len(d.y_test) - 1}, Enter = 0): ", 0, 0, len(d.y_test) - 1)
    x = d.X_test[i].copy()
    base = float(m.predict_proba(x.reshape(1, -1))[0])
    print(f"\nSample {i}: actual = {CLASS_NAMES[d.y_test[i]]}, "
          f"output = {base:.4f}, predicted = {CLASS_NAMES[int(base >= THRESHOLD)]}")
    print(f"Each feature is changed by +/-{PERTURB} (normalised units), one at a time.")
    print("Influence = average |change| of the output neuron's weighted sum (logit). The sigmoid output")
    print("can saturate near 0 or 1, which hides changes, so the output change is shown as well.\n")
    base_logit = float(m.logit(x.reshape(1, -1))[0])
    rows = []
    for j, f in enumerate(FEATURES):
        up, dn = x.copy(), x.copy()
        up[j] += PERTURB
        dn[j] -= PERTURB
        p_up = float(m.predict_proba(up.reshape(1, -1))[0])
        p_dn = float(m.predict_proba(dn.reshape(1, -1))[0])
        l_up = float(m.logit(up.reshape(1, -1))[0])
        l_dn = float(m.logit(dn.reshape(1, -1))[0])
        rows.append({"Feature": f,
                     "Output change (+)": round(p_up - base, 6),
                     "Output change (-)": round(p_dn - base, 6),
                     "Logit change (+)": round(l_up - base_logit, 4),
                     "Logit change (-)": round(l_dn - base_logit, 4),
                     "Influence": round((abs(l_up - base_logit) + abs(l_dn - base_logit)) / 2, 4)})
    df = pd.DataFrame(rows).sort_values("Influence", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.to_string())
    print("\nTop 5 most influential features for this sample:")
    for r, row in df.head(5).iterrows():
        print(f"  {r}. {row['Feature']:<24} influence = {row['Influence']:.4f}")


# ----------------------------------------------------------------------------
# 4. Hidden neuron response analyzer (SHLNN)
# ----------------------------------------------------------------------------
def option4(tr):
    header("4. Hidden Neuron Response Analyzer (SHLNN 30-12-1)")
    d = tr.data
    m = tr.get("SHLNN")
    H = m.hidden_activations(d.X_test)
    n_h = H.shape[1]
    cols = [f"H{k + 1}" for k in range(n_h)]
    n = ask_int(f"Rows of the activation matrix to print (0-{len(H)}, Enter = 20): ", 20, 0, len(H))
    mat = pd.DataFrame(np.round(H, 3), columns=cols)
    mat.insert(0, "Class", [CLASS_NAMES[c][0] for c in d.y_test])
    mat.index.name = "Sample"
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print("\nHidden activation matrix (B = Benign, M = Malignant):")
        print(mat.head(n).to_string())
        avg_b = H[d.y_test == 0].mean(axis=0)
        avg_m = H[d.y_test == 1].mean(axis=0)
        summary = pd.DataFrame({"Avg Benign": np.round(avg_b, 4), "Avg Malignant": np.round(avg_m, 4),
                                "Difference (M-B)": np.round(avg_m - avg_b, 4)}, index=cols)
        print("\nAverage activation per hidden neuron:")
        print(summary.to_string())
    kb, km = int(np.argmax(avg_b)), int(np.argmax(avg_m))
    print(f"\nHighest average activation for BENIGN    : {cols[kb]} ({avg_b[kb]:.4f})")
    print(f"Highest average activation for MALIGNANT : {cols[km]} ({avg_m[km]:.4f})")
    kd = int(np.argmax(np.abs(avg_m - avg_b)))
    print(f"Neuron that best separates the classes   : {cols[kd]} (|difference| = {abs(avg_m[kd] - avg_b[kd]):.4f})")
    print(f"SHLNN test accuracy: {m.accuracy(d.X_test, d.y_test):.2f}%")


# ----------------------------------------------------------------------------
# 5. Input feature masking challenge
# ----------------------------------------------------------------------------
def option5(tr):
    header("5. Input Feature Masking Challenge (BPNN)")
    d = tr.data
    m = tr.get("BPNN")
    choice = input("Mask with 1 = zero, 2 = dataset mean (Enter = 2): ").strip()
    use_zero = choice == "1"
    label = "zero" if use_zero else "dataset mean"
    base = m.accuracy(d.X_test, d.y_test)
    print(f"\nBaseline test accuracy: {base:.2f}%   (masking value: {label})\n")
    rows = []
    for j, f in enumerate(FEATURES):
        Xm = d.X_test.copy()
        Xm[:, j] = 0.0 if use_zero else d.train_mean[j]
        acc = m.accuracy(Xm, d.y_test)
        rows.append({"Masked Feature": f, "Accuracy %": round(acc, 2), "Drop %": round(base - acc, 2)})
    df = pd.DataFrame(rows).sort_values("Drop %", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"
    print(df.to_string())
    worst = df.iloc[0]
    if worst["Drop %"] > 0:
        print(f"\nMaximum performance drop: {worst['Masked Feature']} (accuracy falls by {worst['Drop %']}%)")
    else:
        print("\nMasking a single feature never reduced accuracy: the network is robust to losing "
              "any one input (the information is spread across many correlated features).")


# ----------------------------------------------------------------------------
# 6. Misclassification confidence analyzer
# ----------------------------------------------------------------------------
def option6(tr):
    header("6. Misclassification Confidence Analyzer (BPNN)")
    d = tr.data
    m = tr.get("BPNN")
    out = m.predict_proba(d.X_test)
    pred = (out >= THRESHOLD).astype(int)
    wrong = np.where(pred != d.y_test)[0]
    print(f"Test samples: {len(pred)} | misclassified: {len(wrong)}\n")
    if len(wrong) == 0:
        print("No misclassified test samples - the network classified every test sample correctly.")
        return
    print(f"{'Sample':>6} {'Actual':<10} {'Predicted':<10} {'Output':>8} {'Distance':>9}  Confidence")
    confident_wrong = []
    for i in wrong:
        cat = confidence_category(out[i])
        if cat == "High Confidence":
            confident_wrong.append(i)
        print(f"{i:>6} {CLASS_NAMES[d.y_test[i]]:<10} {CLASS_NAMES[pred[i]]:<10} "
              f"{out[i]:>8.4f} {abs(out[i] - THRESHOLD):>9.4f}  {cat}")
    print()
    if confident_wrong:
        print(f"CONFIDENTLY WRONG predictions (High Confidence but incorrect): {len(confident_wrong)}")
        print("  Samples: " + ", ".join(str(int(i)) for i in confident_wrong))
        print("  The network can be sure and still wrong, so a high sigmoid output alone is not proof of correctness.")
    else:
        print("No confidently wrong predictions: every error lies close to the threshold "
              "(Low/Moderate confidence), so the network 'hesitates' when it is wrong.")


# ----------------------------------------------------------------------------
# 7. Comparative analysis
# ----------------------------------------------------------------------------
def option7(tr):
    header("7. BPNN - MLP - SHLNN Comparative Analysis")
    d = tr.data
    rows = []
    arch = {"BPNN": "30-8-1 (online)", "MLP": "30-16-8-1", "SHLNN": "30-12-1"}
    for kind in ["BPNN", "MLP", "SHLNN"]:
        m = tr.get(kind)
        mt = metrics(d.y_test, m.predict(d.X_test))
        rows.append({"Model": kind, "Architecture": arch[kind], "Parameters": m.n_params(),
                     "Epochs": m.epochs, "Final Loss": round(m.loss_history[-1], 4),
                     "Train Acc %": round(m.accuracy(d.X_train, d.y_train), 2),
                     "Test Acc %": round(mt["acc"], 2), "Precision %": round(mt["prec"], 2),
                     "Recall %": round(mt["rec"], 2), "F1 %": round(mt["f1"], 2),
                     "FP": mt["fp"], "FN": mt["fn"], "Time (s)": round(m.train_time, 2)})
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.to_string(index=False))
    top = df[df["Test Acc %"] == df["Test Acc %"].max()]
    if len(top) > 1:
        print(f"\nTest accuracy is tied at {top['Test Acc %'].iloc[0]}% for: {', '.join(top['Model'])}.")
        best = top.sort_values("Final Loss").iloc[0]
        print(f"Lowest final training loss among them: {best['Model']} ({best['Final Loss']}).")
    else:
        best = top.iloc[0]
        print(f"\nBest test accuracy: {best['Model']} ({best['Test Acc %']}%, F1 {best['F1 %']}%)")
    print("FN (malignant predicted as benign) is the costlier error in cancer screening, "
          "so recall matters as much as accuracy.")


# ----------------------------------------------------------------------------
# Menu
# ----------------------------------------------------------------------------
MENU = """
============ BACKPROPAGATION NEURAL NETWORK MENU ============
 1. Tumor Classification using BPNN
 2. Prediction Confidence Analyzer
 3. Feature Contribution Analyzer
 4. Hidden Neuron Response Analyzer
 5. Input Feature Masking Challenge
 6. Misclassification Confidence Analyzer
 7. BPNN-MLP-SHLNN Comparative Analysis
 8. Exit
=============================================================="""


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    X, y = load_dataset(path)
    data = Data(X, y)
    print(f"Samples: {len(y)} (Benign {int(np.sum(y == 0))}, Malignant {int(np.sum(y == 1))}) | "
          f"features: {X.shape[1]} min-max normalised | train: {len(data.y_train)} | test: {len(data.y_test)}")
    trainer = Trainer(data)
    actions = {"1": option1, "2": option2, "3": option3, "4": option4,
               "5": option5, "6": option6, "7": option7}
    while True:
        print(MENU)
        choice = input("Enter choice (1-8): ").strip()
        if choice == "8":
            print("Goodbye!")
            break
        if choice in actions:
            actions[choice](trainer)
        else:
            print("Invalid choice. Please enter 1-8.")


if __name__ == "__main__":
    main()