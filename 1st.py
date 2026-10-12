"""
Perceptron Decision Boundary and Linear Separability Challenge
Dataset : Banknote Authentication (UCI id=267)
Classes : 0 = genuine, 1 = forged
Usage   : python banknote_perceptron.py [path_to_csv]
          (CSV may be the UCI .txt file without header, or the Kaggle CSV with header)
Needs   : numpy, pandas
"""
import sys
import os
import itertools
import numpy as np
import pandas as pd

FEATURES = ["variance", "skewness", "curtosis", "entropy"]
CLASS_NAMES = {0: "Genuine", 1: "Forged"}
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
SEED = 42
LEARNING_RATE = 0.1
MAX_EPOCHS = 100
TEST_FRACTION = 0.2
BORDERLINE_THRESHOLD = 0.10   # |weighted sum| below this => "borderline" sample


# ----------------------------------------------------------------------------
# Data loading, normalization, split
# ----------------------------------------------------------------------------
def load_dataset(path=None):
    candidates = [path] if path else []
    candidates += ["dataset.csv", "data_banknote_authentication.txt", "BankNote_Authentication.csv",
                   "banknote.csv", "data_banknote_authentication.csv"]
    df = None
    for p in candidates:
        if p and os.path.exists(p):
            df = pd.read_csv(p, header=None)
            print(f"Loaded dataset from: {p}")
            break
    if df is None:
        try:
            df = pd.read_csv(UCI_URL, header=None)
            print("Downloaded dataset from UCI repository.")
        except Exception:
            print("ERROR: dataset not found. Download it from UCI/Kaggle and either place it")
            print("       next to this script or pass its path:  python banknote_perceptron.py <file>")
            sys.exit(1)
    # Kaggle version has a text header row -> drop it
    if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric)
    df.columns = FEATURES + ["class"]
    df["class"] = df["class"].astype(int)
    return df


class Data:
    """Holds raw + min-max normalized train/test data (scaler fitted on train only)."""
    def __init__(self, df):
        X = df[FEATURES].values.astype(float)
        y = df["class"].values.astype(int)
        rng = np.random.default_rng(SEED)
        test_idx = []
        for c in (0, 1):                       # stratified split
            idx = np.where(y == c)[0]
            k = int(round(len(idx) * TEST_FRACTION))
            test_idx += list(rng.choice(idx, size=k, replace=False))
        test_mask = np.zeros(len(y), dtype=bool)
        test_mask[test_idx] = True
        # boolean masks keep the ORIGINAL sample order inside each split
        self.X_train_raw, self.y_train = X[~test_mask], y[~test_mask]
        self.X_test_raw, self.y_test = X[test_mask], y[test_mask]
        self.min = self.X_train_raw.min(axis=0)
        self.max = self.X_train_raw.max(axis=0)
        self.X_train = self.normalize(self.X_train_raw)
        self.X_test = self.normalize(self.X_test_raw)

    def normalize(self, X):
        return (X - self.min) / (self.max - self.min)

    def cols(self, feats):
        return [FEATURES.index(f) for f in feats]


# ----------------------------------------------------------------------------
# Single-Layer Perceptron (step activation, perceptron learning rule)
# ----------------------------------------------------------------------------
class Perceptron:
    def __init__(self, n_inputs, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS):
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.lr = lr
        self.max_epochs = max_epochs
        self.epochs = 0
        self.converged = False
        self.error_history = []

    @staticmethod
    def step(z):
        return (z >= 0).astype(int)

    def weighted_sum(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return self.step(self.weighted_sum(X))

    def fit(self, X, y, order=None):
        """order: array of sample indices defining a fixed presentation order."""
        if order is None:
            order = np.arange(len(y))
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.error_history = []
        self.converged = False
        for epoch in range(1, self.max_epochs + 1):
            errors = 0
            for i in order:
                pred = int(self.w @ X[i] + self.b >= 0)
                err = y[i] - pred
                if err != 0:
                    self.w += self.lr * err * X[i]
                    self.b += self.lr * err
                    errors += 1
            self.error_history.append(errors)
            self.epochs = epoch
            if errors == 0:
                self.converged = True
                break
        return self

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y)) * 100


def train_model(data, feats, order_name="Original"):
    cols = data.cols(feats)
    Xtr = data.X_train[:, cols]
    order = make_order(order_name, data.y_train)
    model = Perceptron(len(cols)).fit(Xtr, data.y_train, order)
    return model


def make_order(name, y):
    n = len(y)
    if name == "Original":
        return np.arange(n)
    if name == "Reverse":
        return np.arange(n)[::-1]
    if name == "Random":
        return np.random.default_rng(SEED).permutation(n)
    if name == "Class-wise grouped":
        return np.concatenate([np.where(y == 0)[0], np.where(y == 1)[0]])
    raise ValueError(name)


def fmt_w(w):
    return "[" + ", ".join(f"{v:.4f}" for v in w) + "]"


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
# 1. Banknote classification
# ----------------------------------------------------------------------------
def option1(data):
    header("1. Banknote Classification using Perceptron (all 4 features)")
    m = train_model(data, FEATURES)
    z = m.weighted_sum(data.X_test)
    pred = m.predict(data.X_test)
    print(f"Epochs used      : {m.epochs}   (converged: {m.converged})")
    print(f"Final weights    : {fmt_w(m.w)}")
    print(f"Final bias       : {m.b:.4f}")
    try:
        n = int(input(f"How many test samples to display? (0-{len(pred)}, Enter = 15): ") or 15)
    except ValueError:
        n = 15
    n = max(0, min(n, len(pred)))
    print(f"\n{'#':>4} {'Actual':<9} {'Predicted':<10} {'Weighted Sum':>13}  Result")
    for i in range(n):
        ok = "OK" if pred[i] == data.y_test[i] else "WRONG"
        print(f"{i:>4} {CLASS_NAMES[data.y_test[i]]:<9} {CLASS_NAMES[pred[i]]:<10} {z[i]:>13.4f}  {ok}")
    print(f"\nTrain accuracy : {m.accuracy(data.X_train, data.y_train):.2f}%")
    print(f"Test accuracy  : {m.accuracy(data.X_test, data.y_test):.2f}%  "
          f"({int(np.sum(pred != data.y_test))} misclassified of {len(pred)})")


# ----------------------------------------------------------------------------
# 2. Decision boundary explorer
# ----------------------------------------------------------------------------
def choose_pair():
    print("Features:", ", ".join(f"{i+1}={f}" for i, f in enumerate(FEATURES)))
    try:
        a = int(input("First feature number : ")) - 1
        b = int(input("Second feature number: ")) - 1
        if a == b or not (0 <= a < 4 and 0 <= b < 4):
            raise ValueError
    except ValueError:
        print("Invalid choice, using variance & skewness.")
        a, b = 0, 1
    return [FEATURES[a], FEATURES[b]]


def option2(data):
    header("2. Decision Boundary Explorer")
    feats = choose_pair()
    m = train_model(data, feats)
    w1, w2 = m.w
    print(f"\nFeatures (normalized): x1={feats[0]}, x2={feats[1]}")
    print(f"Epochs: {m.epochs} (converged: {m.converged})")
    print(f"Weights: w1={w1:.4f}, w2={w2:.4f}   Bias: {m.b:.4f}")
    print(f"Boundary equation: {w1:.4f}*x1 + {w2:.4f}*x2 + ({m.b:.4f}) = 0")
    if abs(w2) > 1e-12:
        print(f"  => x2 = {-w1/w2:.4f}*x1 + {-m.b/w2:.4f}")
    print("Class 1 (Forged) side: weighted sum >= 0 ; Class 0 (Genuine) side: weighted sum < 0")
    cols = data.cols(feats)
    Xt = data.X_test[:, cols]
    z = m.weighted_sum(Xt)
    pred = m.predict(Xt)
    print(f"\n{'#':>4} {'x1':>8} {'x2':>8} {'Sum':>9}  {'Side':<22} {'Actual':<8} Result")
    for i in range(len(z)):
        side = "Forged side (z>=0)" if z[i] >= 0 else "Genuine side (z<0)"
        ok = "OK" if pred[i] == data.y_test[i] else "WRONG"
        print(f"{i:>4} {Xt[i,0]:>8.3f} {Xt[i,1]:>8.3f} {z[i]:>9.4f}  {side:<22} "
              f"{CLASS_NAMES[data.y_test[i]]:<8} {ok}")
    print(f"\nTest accuracy: {m.accuracy(Xt, data.y_test):.2f}%")


# ----------------------------------------------------------------------------
# 3. Feature pair separability
# ----------------------------------------------------------------------------
def pair_results(data):
    rows = []
    for pair in itertools.combinations(FEATURES, 2):
        m = train_model(data, list(pair))
        cols = data.cols(pair)
        tr_pred = m.predict(data.X_train[:, cols])
        rows.append({
            "Feature Pair": " + ".join(pair),
            "Epochs": m.epochs,
            "Converged": "Yes" if m.converged else "No",
            "Train Misclassified": int(np.sum(tr_pred != data.y_train)),
            "Train Acc %": round(m.accuracy(data.X_train[:, cols], data.y_train), 2),
            "Test Acc %": round(m.accuracy(data.X_test[:, cols], data.y_test), 2),
            "Test Misclassified": int(np.sum(m.predict(data.X_test[:, cols]) != data.y_test)),
        })
    df = pd.DataFrame(rows).sort_values(
        ["Train Misclassified", "Epochs", "Test Misclassified"]).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"
    return df


def option3(data):
    header("3. Feature Pair Separability Challenge")
    df = pair_results(data)
    print(df.to_string())
    print(f"\nMost linearly separable pair: {df.iloc[0]['Feature Pair']}")
    print("(ranked by fewest training misclassifications, then fewest epochs)")
    return df


# ----------------------------------------------------------------------------
# 4. Training sample order
# ----------------------------------------------------------------------------
ORDERS = ["Original", "Reverse", "Random", "Class-wise grouped"]


def option4(data):
    header("4. Training Sample Order Challenge (all 4 features)")
    rows, models = [], {}
    for o in ORDERS:
        m = train_model(data, FEATURES, o)
        models[o] = m
        rows.append({"Order": o, "Epochs": m.epochs, "Converged": "Yes" if m.converged else "No",
                     "Weights": fmt_w(m.w), "Bias": round(m.b, 4),
                     "Train Acc %": round(m.accuracy(data.X_train, data.y_train), 2),
                     "Test Acc %": round(m.accuracy(data.X_test, data.y_test), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    base = models["Original"]
    print()
    different = False
    for o in ORDERS[1:]:
        same = np.allclose(base.w, models[o].w) and np.isclose(base.b, models[o].b)
        different |= not same
        print(f"Original vs {o:<19}: boundary {'IDENTICAL' if same else 'DIFFERENT'}")
    if different:
        print("\nConclusion: sample order changes the sequence of weight updates, so different orders"
              "\ngive different (but usually similarly accurate) decision boundaries.")
    else:
        print("\nConclusion: all orders produced the same boundary.")
    return rows


# ----------------------------------------------------------------------------
# 5. Borderline samples
# ----------------------------------------------------------------------------
def borderline_info(data, feats=FEATURES, order="Original"):
    m = train_model(data, feats, order)
    cols = data.cols(feats)
    Xt = data.X_test[:, cols]
    z = m.weighted_sum(Xt)
    return m, cols, Xt, z


def option5(data):
    header("5. Borderline Sample Detector (all 4 features)")
    m, cols, Xt, z = borderline_info(data)
    pred = m.predict(Xt)
    norm = np.linalg.norm(m.w)
    top = np.argsort(np.abs(z))[:5]
    print(f"Weights {fmt_w(m.w)}  Bias {m.b:.4f}\n")
    for rank, i in enumerate(top, 1):
        raw = data.X_test_raw[i]
        print(f"#{rank}  test sample {i}")
        print("    features (raw)       : " + ", ".join(f"{f}={v:.4f}" for f, v in zip(FEATURES, raw)))
        print("    features (normalized): " + ", ".join(f"{v:.4f}" for v in Xt[i]))
        print(f"    |weighted sum|       : {abs(z[i]):.5f}")
        print(f"    distance to boundary : {abs(z[i]) / norm:.5f}   (|z| / ||w||)")
        print(f"    predicted: {CLASS_NAMES[pred[i]]}   actual: {CLASS_NAMES[data.y_test[i]]}")
    n_border = int(np.sum(np.abs(z) < BORDERLINE_THRESHOLD))
    print(f"\nSamples with |weighted sum| < {BORDERLINE_THRESHOLD}: {n_border}")


# ----------------------------------------------------------------------------
# 6. XOR limitation
# ----------------------------------------------------------------------------
GATES = {
    "AND":  [0, 0, 0, 1],
    "OR":   [0, 1, 1, 1],
    "NAND": [1, 1, 1, 0],
    "XOR":  [0, 1, 1, 0],
}
GATE_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def option6(_data=None):
    header("6. XOR Limitation Demonstrator")
    try:
        max_ep = int(input("Maximum epochs (Enter = 50): ") or 50)
    except ValueError:
        max_ep = 50
    summary = []
    for name, t in GATES.items():
        y = np.array(t)
        m = Perceptron(2, lr=0.1, max_epochs=max_ep).fit(GATE_X, y)
        status = "CONVERGED" if m.converged else "DID NOT CONVERGE"
        print(f"\n{name:<5}: {status} in {m.epochs} epoch(s) | w={fmt_w(m.w)} b={m.b:.2f}")
        print(f"       truth: {t}   predicted: {[int(v) for v in m.predict(GATE_X)]}")
        summary.append((name, m.converged, m.epochs))
        if name == "XOR":
            # replay training and record WHICH inputs are misclassified each epoch
            w, b = np.zeros(2), 0.0
            print("\n  XOR misclassification pattern (epoch -> misclassified inputs, errors):")
            patterns = []
            for ep in range(1, max_ep + 1):
                wrong = []
                for i in range(4):
                    p = int(w @ GATE_X[i] + b >= 0)
                    e = y[i] - p
                    if e:
                        w = w + 0.1 * e * GATE_X[i]
                        b += 0.1 * e
                        wrong.append(tuple(int(v) for v in GATE_X[i]))
                patterns.append(wrong)
                if ep <= 8 or ep > max_ep - 2:
                    print(f"    epoch {ep:>3}: {wrong}  ({len(wrong)} error(s))")
                elif ep == 9:
                    print("    ...")
            cycle = [p for p in patterns[-4:]]
            print(f"\n  Errors never reach 0; the pattern keeps repeating (last epochs: "
                  f"{[len(p) for p in cycle]} errors).")
            print("  Why: XOR needs (0,0),(1,1) -> 0 and (0,1),(1,0) -> 1. Any line w1*x1+w2*x2+b=0")
            print("  that puts (0,1) and (1,0) on the positive side also puts the midpoint of those")
            print("  two points on the positive side, and that midpoint (0.5,0.5) is the midpoint of")
            print("  (0,0) and (1,1) too, which must be negative. A single linear boundary cannot do")
            print("  both, so XOR is NOT linearly separable (it needs a hidden layer / MLP).")
    print("\nSummary:")
    for name, c, e in summary:
        print(f"  {name:<5}: {'Linearly separable (converged)' if c else 'NOT linearly separable (no convergence)'}")
    return summary


# ----------------------------------------------------------------------------
# 7. Feature removal
# ----------------------------------------------------------------------------
def feature_removal(data):
    base = train_model(data, FEATURES)
    base_tr = base.accuracy(data.X_train, data.y_train)
    base_te = base.accuracy(data.X_test, data.y_test)
    rows = []
    for f in FEATURES:
        rest = [x for x in FEATURES if x != f]
        m = train_model(data, rest)
        cols = data.cols(rest)
        tr = m.accuracy(data.X_train[:, cols], data.y_train)
        te = m.accuracy(data.X_test[:, cols], data.y_test)
        rows.append({"Removed": f, "Epochs": m.epochs, "Converged": "Yes" if m.converged else "No",
                     "Train Acc %": round(tr, 2), "Test Acc %": round(te, 2),
                     "Train Drop": round(base_tr - tr, 2), "Test Drop": round(base_te - te, 2),
                     "_model": m, "_feats": rest})
    return base, base_tr, base_te, rows


def option7(data):
    header("7. Perceptron Feature Removal Challenge")
    base, btr, bte, rows = feature_removal(data)
    print(f"Baseline (all features): train {btr:.2f}%  test {bte:.2f}%  epochs {base.epochs}\n")
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    print(df.to_string(index=False))
    worst = df.sort_values(["Test Drop", "Train Drop"], ascending=False).iloc[0]
    print(f"\nLargest accuracy drop when removing: {worst['Removed']} "
          f"(test drop {worst['Test Drop']}%, train drop {worst['Train Drop']}%)")
    print("=> this feature contributes most to separating genuine from forged notes.")


# ----------------------------------------------------------------------------
# 8. Comparative report
# ----------------------------------------------------------------------------
def option8(data):
    header("8. Perceptron Classification Report (running all experiments)")
    experiments = []   # (label, feats, order)
    experiments.append(("Exp1: All features", FEATURES, "Original"))
    pairs = pair_results(data)
    best_pair = pairs.iloc[0]["Feature Pair"].split(" + ")
    experiments.append(("Exp3: Best feature pair", best_pair, "Original"))
    for o in ORDERS:
        experiments.append((f"Exp4: Order={o}", FEATURES, o))
    _, _, _, removal = feature_removal(data)
    for r in removal:
        experiments.append((f"Exp7: Without {r['Removed']}", r["_feats"], "Original"))

    rows = []
    for label, feats, order in experiments:
        m = train_model(data, feats, order)
        cols = data.cols(feats)
        z = m.weighted_sum(data.X_test[:, cols])
        rows.append({
            "Experiment": label,
            "Features": ",".join(feats),
            "Order": order,
            "Epochs": m.epochs,
            "Weights": fmt_w(m.w),
            "Bias": round(m.b, 4),
            "Train Acc %": round(m.accuracy(data.X_train[:, cols], data.y_train), 2),
            "Test Acc %": round(m.accuracy(data.X_test[:, cols], data.y_test), 2),
            "Borderline": int(np.sum(np.abs(z) < BORDERLINE_THRESHOLD)),
            "Converged": "Yes" if m.converged else "No",
        })
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 250,
                           "display.max_colwidth", 60):
        print(df.to_string(index=False))
    print(f"\nBorderline = test samples with |weighted sum| < {BORDERLINE_THRESHOLD}")
    xor_ok = []
    for name, t in GATES.items():
        m = Perceptron(2, max_epochs=50).fit(GATE_X, np.array(t))
        xor_ok.append(f"{name}: {'converged' if m.converged else 'did not converge'} ({m.epochs} ep)")
    print("Logic gates (Exp6): " + " | ".join(xor_ok))
    return df


# ----------------------------------------------------------------------------
# Menu
# ----------------------------------------------------------------------------
MENU = """
==================== PERCEPTRON CHALLENGE MENU ====================
 1. Banknote Classification using Perceptron
 2. Decision Boundary Explorer
 3. Feature Pair Separability Challenge
 4. Training Sample Order Challenge
 5. Borderline Sample Detector
 6. XOR Limitation Demonstrator
 7. Perceptron Feature Removal Challenge
 8. Perceptron Classification Report
 9. Exit
===================================================================="""


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_dataset(path)
    data = Data(df)
    print(f"Samples: {len(df)} | train: {len(data.y_train)} | test: {len(data.y_test)} | "
          f"features min-max normalized to [0,1] | classes: 0=Genuine, 1=Forged")
    actions = {"1": option1, "2": option2, "3": option3, "4": option4,
               "5": option5, "6": option6, "7": option7, "8": option8}
    while True:
        print(MENU)
        choice = input("Enter choice (1-9): ").strip()
        if choice == "9":
            print("Goodbye!")
            break
        if choice in actions:
            actions[choice](data)
        else:
            print("Invalid choice. Please enter 1-9.")


if __name__ == "__main__":
    main()