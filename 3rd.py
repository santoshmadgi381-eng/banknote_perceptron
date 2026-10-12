import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import copy

class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=50):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.costs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.costs = []

        for _ in range(self.n_iterations):
            net_input = self.activation(X)
            errors = y - net_input
            
            # Delta Rule / Gradient Descent weight update
            self.weights += self.learning_rate * X.T.dot(errors) / n_samples
            self.bias += self.learning_rate * errors.mean()
            
            # Calculate Mean Squared Error
            cost = (errors**2).mean() / 2.0
            self.costs.append(cost)
        return self

    def activation(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        # Threshold at 0.5 since classes are 0 and 1
        return np.where(self.activation(X) >= 0.5, 1, 0)

def load_and_prep_data():
    # Load dataset directly from the UCI Machine Learning Repository URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    print(f"Loading dataset from: {url}")
    
    feature_names = [
        'sepal length (cm)', 
        'sepal width (cm)', 
        'petal length (cm)', 
        'petal width (cm)'
    ]
    
    df = pd.read_csv(url, header=None, names=feature_names + ['class'])
    
    # Filter for Setosa and Versicolor only
    df = df[df['class'].isin(['Iris-setosa', 'Iris-versicolor'])]
    
    # Encode target classes: Iris-setosa = 0, Iris-versicolor = 1
    df['class'] = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})
    
    X = df[feature_names].values
    y = df['class'].values
    
    return train_test_split(X, y, test_size=0.3, random_state=42), feature_names

def menu_1(X_train, X_test, y_train, y_test):
    print("\n--- 1. Binary Pattern Classification ---")
    model = Adaline(learning_rate=0.01, n_iterations=100)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    linear_outputs = model.activation(X_test)
    acc = accuracy_score(y_test, predictions)
    
    df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions,
        'Linear Output': np.round(linear_outputs, 4)
    })
    print(df.head(10).to_string())
    print(f"\nFinal Classification Accuracy: {acc * 100:.2f}%")
    return model

def menu_2(model, feature_names):
    print("\n--- 2. Multi-Input ADALINE Feature Prediction ---")
    print("Enter values for a new flower sample:")
    try:
        sl = float(input(f"{feature_names[0]}: "))
        sw = float(input(f"{feature_names[1]}: "))
        pl = float(input(f"{feature_names[2]}: "))
        pw = float(input(f"{feature_names[3]}: "))
        
        sample = np.array([[sl, sw, pl, pw]])
        linear_output = model.activation(sample)[0]
        prediction = model.predict(sample)[0]
        
        print(f"\nLinear Output: {linear_output:.4f}")
        print(f"Predicted Class: {'Iris Versicolor (1)' if prediction == 1 else 'Iris Setosa (0)'}")
        print("\nLearned Weights:")
        for name, weight in zip(feature_names, model.weights):
            print(f" - {name}: {weight:.4f}")
        print(f" - Bias: {model.bias:.4f}")
    except ValueError:
        print("Invalid input. Please enter numerical values.")

def menu_3(model, X_test, feature_names):
    print("\n--- 3. Feature Influence Analyzer ---")
    sample = X_test[0].copy()
    base_output = model.activation(sample.reshape(1, -1))[0]
    
    print(f"Baseline Linear Output: {base_output:.4f}")
    influences = []
    
    # Increase each feature by 10% to measure influence
    for i in range(len(sample)):
        modified_sample = sample.copy()
        modified_sample[i] *= 1.10  
        new_output = model.activation(modified_sample.reshape(1, -1))[0]
        change = abs(new_output - base_output)
        influences.append((feature_names[i], change))
        
    influences.sort(key=lambda x: x[1], reverse=True)
    print("\nFeature Influence Ranking (based on 10% perturbation):")
    for rank, (feat, change) in enumerate(influences, 1):
        print(f"{rank}. {feat} -> Output Change: {change:.6f}")
        
    print(f"\nMost influential feature: {influences[0][0]}")

def menu_4(model, X_test, y_test):
    print("\n--- 4. Noise Filtering Challenge ---")
    base_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Original Accuracy: {base_acc * 100:.2f}%")
    
    noise_levels = [0.05, 0.10, 0.20, 0.30]
    max_tolerated = 0
    
    for noise in noise_levels:
        np.random.seed(42)
        # Add random uniform noise scaled by the feature values
        noise_matrix = np.random.uniform(-noise, noise, X_test.shape)
        X_test_noisy = X_test * (1 + noise_matrix)
        
        preds = model.predict(X_test_noisy)
        acc = accuracy_score(y_test, preds)
        changed_preds = np.sum(preds != model.predict(X_test))
        
        print(f"\nNoise Level: {noise * 100}%")
        print(f"Noisy-data Accuracy: {acc * 100:.2f}%")
        print(f"Changed Predictions: {changed_preds} out of {len(y_test)}")
        
        if acc == base_acc:
            max_tolerated = noise * 100
            
    print(f"\nMaximum Noise Level Tolerated (without accuracy drop): {max_tolerated}%")

def menu_5(model, X_test, y_test):
    print("\n--- 5. Misclassification Pattern Analyzer ---")
    preds = model.predict(X_test)
    linear_outputs = model.activation(X_test)
    
    misclassified_idx = np.where(preds != y_test)[0]
    
    if len(misclassified_idx) == 0:
        print("Model achieved 100% accuracy. No misclassified samples found.")
        print("Note: Iris Setosa and Versicolor are completely linearly separable.")
    else:
        for idx in misclassified_idx:
            dist = abs(linear_outputs[idx] - 0.5)
            print(f"Sample {idx}:")
            print(f"  Features: {X_test[idx]}")
            print(f"  Actual: {y_test[idx]}, Predicted: {preds[idx]}")
            print(f"  Linear Output: {linear_outputs[idx]:.4f}")
            print(f"  Distance from Threshold (0.5): {dist:.4f}\n")

def menu_6(X_train, X_test, y_train, feature_names):
    print("\n--- 6. Decision Boundary Analyzer ---")
    # Using Sepal Length (0) and Petal Length (2) for visualization purposes
    idx1, idx2 = 0, 2
    print(f"Selected Features: {feature_names[idx1]} and {feature_names[idx2]}")
    
    X_train_2d = X_train[:, [idx1, idx2]]
    X_test_2d = X_test[:, [idx1, idx2]]
    
    model_2d = Adaline(learning_rate=0.01, n_iterations=100)
    model_2d.fit(X_train_2d, y_train)
    
    w1, w2 = model_2d.weights
    b = model_2d.bias
    
    print(f"Decision Boundary Equation: {w1:.4f}*(F1) + {w2:.4f}*(F2) + {b:.4f} = 0.5")
    
    preds = model_2d.predict(X_test_2d)
    setosa_count = np.sum(preds == 0)
    versicolor_count = np.sum(preds == 1)
    
    print(f"\nTest Samples Classification:")
    print(f"Predicted Iris Setosa (0): {setosa_count}")
    print(f"Predicted Iris Versicolor (1): {versicolor_count}")

def menu_7(X_train, X_test, y_train, y_test, feature_names):
    print("\n--- 7. Comparative Analysis ---")
    print("Executing combinations of features to find the best configuration...")
    
    results = []
    
    # Test Full features vs 2-feature sets
    configs = [
        ("All Features", [0, 1, 2, 3]),
        ("Sepal Features", [0, 1]),
        ("Petal Features", [2, 3])
    ]
    
    for name, indices in configs:
        X_tr = X_train[:, indices]
        X_te = X_test[:, indices]
        
        m = Adaline(learning_rate=0.01, n_iterations=100).fit(X_tr, y_train)
        preds = m.predict(X_te)
        acc = accuracy_score(y_test, preds)
        mse = m.costs[-1]
        misclassified = np.sum(preds != y_test)
        
        results.append({
            "Config": name,
            "Inputs": len(indices),
            "Final MSE": mse,
            "Accuracy": acc,
            "Misclassified": misclassified
        })
        
    df_results = pd.DataFrame(results)
    print("\nComparative Report:")
    print(df_results.to_string(index=False))
    
    best_config = df_results.sort_values(by=['Accuracy', 'Final MSE'], ascending=[False, True]).iloc[0]
    print(f"\nBest Feature Combination: {best_config['Config']} with {best_config['Accuracy']*100}% accuracy and MSE of {best_config['Final MSE']:.4f}")

def main():
    try:
        (X_train, X_test, y_train, y_test), feature_names = load_and_prep_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have an active internet connection to download the dataset.")
        return

    global_model = None

    while True:
        print("\n" + "="*50)
        print("   ADALINE Adaptive Pattern Classification System   ")
        print("="*50)
        print("1. Binary Pattern Classification")
        print("2. Multi-Input ADALINE Feature Prediction")
        print("3. Feature Influence Analyzer")
        print("4. Noise Filtering Challenge")
        print("5. Misclassification Pattern Analyzer")
        print("6. Decision Boundary Analyzer")
        print("7. Comparative Analysis")
        print("8. Exit")
        
        choice = input("Enter your choice (1-8): ")
        
        if choice == '1':
            global_model = menu_1(X_train, X_test, y_train, y_test)
        elif choice == '2':
            if not global_model:
                print("Please run Option 1 first to train the baseline model.")
            else:
                menu_2(global_model, feature_names)
        elif choice == '3':
            if not global_model:
                print("Please run Option 1 first to train the baseline model.")
            else:
                menu_3(global_model, X_test, feature_names)
        elif choice == '4':
            if not global_model:
                print("Please run Option 1 first to train the baseline model.")
            else:
                menu_4(global_model, X_test, y_test)
        elif choice == '5':
            if not global_model:
                print("Please run Option 1 first to train the baseline model.")
            else:
                menu_5(global_model, X_test, y_test)
        elif choice == '6':
            menu_6(X_train, X_test, y_train, feature_names)
        elif choice == '7':
            menu_7(X_train, X_test, y_train, y_test, feature_names)
        elif choice == '8':
            print("Exiting ADALINE System. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 8.")

if __name__ == "__main__":
    main()