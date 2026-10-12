import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import copy

class Madaline:
    def __init__(self, n_units=3, learning_rate=0.01, n_iterations=100):
        self.n_units = n_units
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights randomly between -0.5 and 0.5
        np.random.seed(42)
        self.weights = np.random.uniform(-0.5, 0.5, (self.n_units, n_features))
        self.bias = np.random.uniform(-0.5, 0.5, self.n_units)

        # Map target y from 0/1 to -1/1 for MADALINE training
        y_bipolar = np.where(y == 0, -1, 1)

        for epoch in range(self.n_iterations):
            for i in range(n_samples):
                xi = X[i]
                target = y_bipolar[i]
                
                # Calculate linear outputs and activations
                z = np.dot(self.weights, xi) + self.bias
                activations = np.where(z >= 0, 1, -1)
                
                # Majority vote
                majority_vote = 1 if np.sum(activations) >= 0 else -1
                
                # Minimal Disturbance Principle (MADALINE Rule I)
                if majority_vote != target:
                    if target == 1:
                        # Find the unit closest to zero and make it positive
                        closest_idx = np.argmin(np.abs(z))
                        self.weights[closest_idx] += self.learning_rate * (1 - z[closest_idx]) * xi
                        self.bias[closest_idx] += self.learning_rate * (1 - z[closest_idx])
                    else:
                        # Find the unit closest to zero and make it negative
                        closest_idx = np.argmin(np.abs(z))
                        self.weights[closest_idx] += self.learning_rate * (-1 - z[closest_idx]) * xi
                        self.bias[closest_idx] += self.learning_rate * (-1 - z[closest_idx])
        return self

    def get_unit_outputs(self, X):
        z = np.dot(X, self.weights.T) + self.bias
        return np.where(z >= 0, 1, -1), z

    def predict(self, X):
        activations, _ = self.get_unit_outputs(X)
        majority = np.where(np.sum(activations, axis=1) >= 0, 1, -1)
        # Map back to 0 and 1
        return np.where(majority == 1, 1, 0)

def load_and_prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    print(f"Loading dataset from: {url}")
    
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv(url, names=columns, na_values='?')
    df = df.dropna()
    
    # 0 = Absence, 1-4 = Presence. Convert to binary 0 and 1
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Normalize inputs (Crucial for MADALINE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42), df.columns[:-1], scaler

def menu_1(X_train, X_test, y_train, y_test):
    print("\n--- 1. Heart Disease Classification using MADALINE ---")
    model = Madaline(n_units=5, learning_rate=0.01, n_iterations=100)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    unit_outputs, _ = model.get_unit_outputs(X_test)
    acc = accuracy_score(y_test, preds)
    
    df_res = pd.DataFrame({
        'Actual': y_test,
        'Predicted': preds,
        'Unit Decisions': [list(row) for row in unit_outputs]
    })
    
    print(df_res.head(10).to_string())
    print(f"\nOverall Classification Accuracy: {acc * 100:.2f}%")
    return model

def menu_2(model, X_test):
    print("\n--- 2. ADALINE Unit Voting Analyzer ---")
    unit_outputs, _ = model.get_unit_outputs(X_test)
    
    unanimous = 0
    majority = 0
    
    print(f"{'Record':<8} {'Votes (+1, -1)':<15} {'Decision Type'}")
    print("-" * 40)
    
    for i, votes in enumerate(unit_outputs):
        pos_votes = np.sum(votes == 1)
        neg_votes = np.sum(votes == -1)
        
        if pos_votes == model.n_units or neg_votes == model.n_units:
            decision = "Unanimous"
            unanimous += 1
        elif abs(pos_votes - neg_votes) <= 1:
            decision = "Tie / Near-Tie"
            majority += 1
        else:
            decision = "Majority"
            majority += 1
            
        if i < 15: # Print first 15 for brevity
            print(f"{i:<8} +1: {pos_votes}, -1: {neg_votes}    {decision}")
            
    print("\nSummary:")
    total = len(X_test)
    print(f"Unanimous Decisions: {unanimous} ({(unanimous/total)*100:.2f}%)")
    print(f"Majority/Tie Decisions: {majority} ({(majority/total)*100:.2f}%)")

def menu_3(model, X_train, y_train):
    print("\n--- 3. Minimal Disturbance Weight Update Challenge ---")
    # Map target to -1/1
    y_bipolar = np.where(y_train == 0, -1, 1)
    
    # Find a misclassified sample
    preds = np.where(model.predict(X_train) == 0, -1, 1)
    misclassified_idx = np.where(preds != y_bipolar)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassified samples found in training set.")
        return
        
    idx = misclassified_idx[0]
    sample = X_train[idx]
    target = y_bipolar[idx]
    
    unit_outputs, z = model.get_unit_outputs(sample.reshape(1, -1))
    z = z[0]
    
    print(f"Target: {target}, Current Prediction: {preds[idx]}")
    print(f"Linear outputs (Z) of ADALINE units: {np.round(z, 4)}")
    
    closest_idx = np.argmin(np.abs(z))
    print(f"Unit closest to decision threshold (Minimal Disturbance): Unit {closest_idx + 1}")
    
    old_weights = model.weights[closest_idx].copy()
    
    # Apply minimal update
    lr = 0.5
    if target == 1:
        model.weights[closest_idx] += lr * (1 - z[closest_idx]) * sample
        model.bias[closest_idx] += lr * (1 - z[closest_idx])
    else:
        model.weights[closest_idx] += lr * (-1 - z[closest_idx]) * sample
        model.bias[closest_idx] += lr * (-1 - z[closest_idx])
        
    new_z = np.dot(model.weights, sample) + model.bias
    new_pred = 1 if np.sum(np.where(new_z >= 0, 1, -1)) >= 0 else -1
    
    print("\nUpdate Applied.")
    print(f"Old Weights (Sample): {np.round(old_weights[:3], 4)}...")
    print(f"New Weights (Sample): {np.round(model.weights[closest_idx][:3], 4)}...")
    print(f"New Linear outputs (Z): {np.round(new_z, 4)}")
    print(f"New MADALINE Decision: {new_pred} (Target was {target})")

def menu_4(model, X_test, feature_names):
    print("\n--- 4. Hidden ADALINE Unit Disagreement Analyzer ---")
    unit_outputs, _ = model.get_unit_outputs(X_test)
    
    disagreements = []
    for i, votes in enumerate(unit_outputs):
        pos_votes = np.sum(votes == 1)
        neg_votes = np.sum(votes == -1)
        minority = min(pos_votes, neg_votes)
        disagreement_pct = (minority / model.n_units) * 100
        disagreements.append((i, disagreement_pct, X_test[i]))
        
    disagreements.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 5 Records with Highest Disagreement among Units:")
    for rank, (idx, pct, features) in enumerate(disagreements[:5], 1):
        print(f"\n{rank}. Record {idx} - Disagreement: {pct:.2f}%")
        # Show top 3 most extreme features for this record
        feat_mags = np.abs(features)
        top_feats = np.argsort(feat_mags)[-3:]
        print("  Key conflicting attributes (highest normalized magnitude):")
        for f_idx in top_feats:
            print(f"    - {feature_names[f_idx]}: {features[f_idx]:.4f}")

def menu_5(model, X_test, feature_names):
    print("\n--- 5. Patient Decision Flip Analyzer ---")
    sample_idx = 0
    sample = X_test[sample_idx].copy()
    feat_idx = 0 # Modify the first feature ('age')
    
    base_pred = model.predict(sample.reshape(1, -1))[0]
    print(f"Analyzing Record {sample_idx}, Base Prediction: {base_pred}")
    print(f"Modifying attribute: {feature_names[feat_idx]}")
    
    flip_val = None
    # Sweep feature from -3.0 to 3.0 (normalized standard deviations)
    for val in np.linspace(-3.0, 3.0, 60):
        test_sample = sample.copy()
        test_sample[feat_idx] = val
        
        pred = model.predict(test_sample.reshape(1, -1))[0]
        if pred != base_pred:
            flip_val = val
            break
            
    if flip_val is not None:
        print(f"Decision flips from Class {base_pred} to Class {pred} at normalized value: {flip_val:.4f}")
    else:
        print("No decision flip observed within +/- 3.0 standard deviations for this feature.")

def menu_6(X_train, X_test, y_train, y_test):
    print("\n--- 6. MADALINE Network Size Challenge ---")
    sizes = [3, 5, 7, 9, 11]
    
    for size in sizes:
        m = Madaline(n_units=size, learning_rate=0.01, n_iterations=100)
        m.fit(X_train, y_train)
        
        preds = m.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        unit_outputs, _ = m.get_unit_outputs(X_test)
        unanimous = sum([1 for v in unit_outputs if abs(np.sum(v)) == size])
        misclassified = np.sum(preds != y_test)
        
        print(f"\nNetwork Size: {size} Units")
        print(f"  Training Epochs: 100")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(f"  Unanimous Decisions: {unanimous}")
        print(f"  Conflicting Decisions: {len(X_test) - unanimous}")
        print(f"  Misclassified: {misclassified}")

def menu_7(model, X_test):
    print("\n--- 7. Ambiguous Patient Record Detector ---")
    unit_outputs, _ = model.get_unit_outputs(X_test)
    
    margins = []
    for i, votes in enumerate(unit_outputs):
        pos = np.sum(votes == 1)
        neg = np.sum(votes == -1)
        margin = abs(pos - neg)
        margins.append((i, pos, neg, margin))
        
    margins.sort(key=lambda x: x[3]) # Sort by lowest margin
    
    print("Top 5 Most Ambiguous Records (Smallest Vote Margin):")
    for rank, (idx, pos, neg, margin) in enumerate(margins[:5], 1):
        print(f"{rank}. Record {idx} | Positive Votes: {pos} | Negative Votes: {neg} | Margin: {margin}")

def menu_8(X_train, X_test, y_train, y_test):
    print("\n--- 8. MADALINE Classification Performance Report ---")
    sizes = [3, 5, 7]
    report = []
    
    for size in sizes:
        m = Madaline(n_units=size, learning_rate=0.01, n_iterations=100)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        
        unit_outputs, _ = m.get_unit_outputs(X_test)
        unanimous = sum([1 for v in unit_outputs if abs(np.sum(v)) == size])
        acc = accuracy_score(y_test, preds)
        
        report.append({
            "Units": size,
            "Accuracy": f"{acc*100:.2f}%",
            "Unanimous": unanimous,
            "Conflicting": len(X_test) - unanimous,
            "Misclassified": np.sum(preds != y_test)
        })
        
    df = pd.DataFrame(report)
    print(df.to_string(index=False))

def main():
    try:
        (X_train, X_test, y_train, y_test), feature_names, scaler = load_and_prep_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    global_model = None

    while True:
        print("\n" + "="*55)
        print(" MADALINE Heart Disease Adaptive Decision Analysis ")
        print("="*55)
        print("1. Heart Disease Classification using MADALINE")
        print("2. ADALINE Unit Voting Analyzer")
        print("3. Minimal Disturbance Weight Update Challenge")
        print("4. Hidden ADALINE Unit Disagreement Analyzer")
        print("5. Patient Decision Flip Analyzer")
        print("6. MADALINE Network Size Challenge")
        print("7. Ambiguous Patient Record Detector")
        print("8. Classification Performance Report")
        print("9. Exit")
        
        choice = input("Enter choice (1-9): ")
        
        if choice == '1':
            global_model = menu_1(X_train, X_test, y_train, y_test)
        elif choice == '2':
            if global_model: menu_2(global_model, X_test)
            else: print("Run option 1 first.")
        elif choice == '3':
            if global_model: menu_3(global_model, X_train, y_train)
            else: print("Run option 1 first.")
        elif choice == '4':
            if global_model: menu_4(global_model, X_test, feature_names)
            else: print("Run option 1 first.")
        elif choice == '5':
            if global_model: menu_5(global_model, X_test, feature_names)
            else: print("Run option 1 first.")
        elif choice == '6':
            menu_6(X_train, X_test, y_train, y_test)
        elif choice == '7':
            if global_model: menu_7(global_model, X_test)
            else: print("Run option 1 first.")
        elif choice == '8':
            menu_8(X_train, X_test, y_train, y_test)
        elif choice == '9':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()