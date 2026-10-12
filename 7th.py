import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

class CompetitiveLearningSOM:
    def __init__(self, n_neurons=5, learning_rate=0.1, n_iterations=100, random_state=42):
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize weights randomly within the range of the data
        min_vals, max_vals = X.min(axis=0), X.max(axis=0)
        self.weights = np.random.uniform(min_vals, max_vals, (self.n_neurons, n_features))
        
        for epoch in range(self.n_iterations):
            # Learning rate decay
            lr = self.learning_rate * (1 - epoch / self.n_iterations)
            
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            for i in indices:
                x = X[i]
                # Find Best Matching Unit (BMU)
                distances = np.linalg.norm(self.weights - x, axis=1)
                bmu_idx = np.argmin(distances)
                
                # Update winning neuron's weight (Competitive Learning)
                self.weights[bmu_idx] += lr * (x - self.weights[bmu_idx])
        return self

    def get_distances(self, x):
        return np.linalg.norm(self.weights - x, axis=1)

    def predict(self, X):
        predictions = []
        for x in X:
            bmu_idx = np.argmin(self.get_distances(x))
            predictions.append(bmu_idx)
        return np.array(predictions)
        
    def quantization_error(self, X):
        error = 0
        for x in X:
            bmu_idx = np.argmin(self.get_distances(x))
            error += np.linalg.norm(x - self.weights[bmu_idx])
        return error / len(X)

def load_and_prep_data():
    # Loading standard Mall Customers dataset from a reliable public raw repository
    url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"
    print(f"Loading dataset from: {url}...")
    
    df = pd.read_csv(url)
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X_raw = df[features].values
    customer_ids = df['CustomerID'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    return X_scaled, X_raw, customer_ids, features, scaler

def menu_1(X_scaled, X_raw):
    print("\n--- 1. Customer Segmentation using SOM ---")
    try:
        n_neurons = int(input("Enter number of SOM neurons (clusters): "))
    except ValueError:
        print("Invalid input. Defaulting to 5.")
        n_neurons = 5
        
    som = CompetitiveLearningSOM(n_neurons=n_neurons, learning_rate=0.1, n_iterations=100)
    som.fit(X_scaled)
    clusters = som.predict(X_scaled)
    
    print("\nCluster Assignments Overview:")
    for c in range(n_neurons):
        cluster_customers = np.where(clusters == c)[0]
        print(f"Cluster {c}: {len(cluster_customers)} customers assigned.")
        if len(cluster_customers) > 0:
            sample_ids = cluster_customers[:5]
            print(f"  -> Sample Customer indices: {sample_ids}")
            
    return som

def menu_2(som, X_scaled):
    print("\n--- 2. Best Matching Unit Explorer ---")
    try:
        idx = int(input(f"Enter customer index (0 to {len(X_scaled)-1}): "))
        if idx < 0 or idx >= len(X_scaled):
            raise ValueError
    except ValueError:
        print("Invalid index. Using index 0.")
        idx = 0
        
    x = X_scaled[idx]
    distances = som.get_distances(x)
    bmu_idx = np.argmin(distances)
    
    print(f"\nCustomer Index: {idx}")
    print("Euclidean distance to each SOM neuron:")
    for i, dist in enumerate(distances):
        print(f"  Neuron {i}: {dist:.4f}")
        
    print(f"\nBest Matching Unit (BMU): Neuron {bmu_idx}")
    print(f"Winning Cluster Assigned: Cluster {bmu_idx}")

def menu_3(X_scaled):
    print("\n--- 3. Dynamic Cluster Count Challenge ---")
    configs = [2, 3, 4, 5, 6]
    best_qe = float('inf')
    best_config = None
    
    for n in configs:
        som = CompetitiveLearningSOM(n_neurons=n, learning_rate=0.1, n_iterations=100)
        som.fit(X_scaled)
        preds = som.predict(X_scaled)
        qe = som.quantization_error(X_scaled)
        
        counts = [np.sum(preds == c) for c in range(n)]
        empty_clusters = counts.count(0)
        imbalanced = max(counts) > (len(X_scaled) / n) * 2 # Simple heuristic for imbalance
        
        print(f"\nConfiguration: {n} Neurons")
        print(f"  Cluster Distribution: {counts}")
        print(f"  Empty Clusters: {empty_clusters}")
        print(f"  Highly Imbalanced: {'Yes' if imbalanced else 'No'}")
        print(f"  Quantization Error: {qe:.4f}")
        
        if qe < best_qe and empty_clusters == 0:
            best_qe = qe
            best_config = n
            
    print(f"\nMost suitable number of neurons (lowest error with no empty clusters): {best_config}")

def menu_4(som, X_scaled, scaler, features):
    print("\n--- 4. Customer Cluster Migration Analyzer ---")
    idx = 0 # Using first customer for demonstration
    x_orig = X_scaled[idx].copy()
    base_bmu = np.argmin(som.get_distances(x_orig))
    
    print("Select feature to modify:")
    print(f"1. {features[1]} (Income)")
    print(f"2. {features[2]} (Spending)")
    choice = input("Choice (1 or 2): ")
    
    feat_idx = 1 if choice == '1' else 2
    feat_name = features[feat_idx]
    
    print(f"\nAnalyzing Record {idx}, Original Cluster: {base_bmu}")
    
    # Sweep through a range of normalized values
    migration_val = None
    new_cluster = None
    
    for val_scaled in np.linspace(-3.0, 3.0, 50):
        x_mod = x_orig.copy()
        x_mod[feat_idx] = val_scaled
        
        bmu = np.argmin(som.get_distances(x_mod))
        if bmu != base_bmu:
            # Unscale to get the real feature value
            x_unscaled = scaler.inverse_transform([x_mod])[0]
            migration_val = x_unscaled[feat_idx]
            new_cluster = bmu
            break
            
    if migration_val:
        print(f"Customer migrates to Cluster {new_cluster} when {feat_name} reaches ~{migration_val:.2f}")
    else:
        print(f"No cluster migration observed within +/- 3 std deviations of {feat_name}.")

def generate_description(age, income, spending):
    desc = []
    desc.append("Young" if age < 35 else "Middle-Aged" if age < 55 else "Senior")
    desc.append("High-Income" if income > 75 else "Moderate-Income" if income > 40 else "Low-Income")
    desc.append("High-Spending" if spending > 70 else "Average-Spending" if spending > 40 else "Low-Spending")
    return f"{desc[0]} {desc[1]} {desc[2]} Customers"

def menu_5(som, scaler):
    print("\n--- 5. Cluster Prototype Generator ---")
    # Inverse transform weights to original feature space
    prototypes = scaler.inverse_transform(som.weights)
    
    for i, proto in enumerate(prototypes):
        age, income, spending = proto
        description = generate_description(age, income, spending)
        
        print(f"\nCluster {i} Prototype:")
        print(f"  Age: {age:.1f}")
        print(f"  Annual Income: ${income*1000:,.0f}")
        print(f"  Spending Score: {spending:.1f}")
        print(f"  -> Generated Profile: {description}")

def menu_6(som, X_scaled, X_raw, customer_ids):
    print("\n--- 6. Outlier Customer Detector ---")
    outliers = []
    
    for i, x in enumerate(X_scaled):
        distances = som.get_distances(x)
        bmu_idx = np.argmin(distances)
        bmu_dist = distances[bmu_idx]
        outliers.append((customer_ids[i], X_raw[i], bmu_idx, bmu_dist))
        
    # Sort by BMU distance descending
    outliers.sort(key=lambda x: x[3], reverse=True)
    
    print("Top 5 Most Unusual Customers (Potential Outliers):")
    for rank, (cid, features, bmu, dist) in enumerate(outliers[:5], 1):
        print(f"\n{rank}. Customer ID: {cid}")
        print(f"   Features (Age, Income, Spend): {features}")
        print(f"   Assigned Cluster: {bmu}")
        print(f"   BMU Distance: {dist:.4f}")

def menu_7(X_scaled):
    print("\n--- 7. SOM Cluster Quality Comparison ---")
    configs = [
        (3, 0.1), (5, 0.1), (7, 0.1),
        (5, 0.01), (5, 0.5)
    ]
    
    results = []
    for n, lr in configs:
        som = CompetitiveLearningSOM(n_neurons=n, learning_rate=lr, n_iterations=100)
        som.fit(X_scaled)
        preds = som.predict(X_scaled)
        qe = som.quantization_error(X_scaled)
        
        counts = [np.sum(preds == c) for c in range(n)]
        empty = counts.count(0)
        max_size = max(counts)
        min_size = min(counts)
        
        results.append({
            "Neurons": n,
            "Learning Rate": lr,
            "Iterations": 100,
            "Empty Clusters": empty,
            "Quant Error": round(qe, 4),
            "Max Size": max_size,
            "Min Size": min_size
        })
        
    df = pd.DataFrame(results)
    print("\nSOM Quality Comparative Report:")
    print(df.to_string(index=False))

def main():
    try:
        X_scaled, X_raw, customer_ids, features, scaler = load_and_prep_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    global_som = None

    while True:
        print("\n" + "="*55)
        print(" SOM Customer Segmentation & Exploration System ")
        print("="*55)
        print("1. Customer Segmentation using SOM")
        print("2. Best Matching Unit Explorer")
        print("3. Dynamic Cluster Count Challenge")
        print("4. Customer Cluster Migration Analyzer")
        print("5. Cluster Prototype Generator")
        print("6. Outlier Customer Detector")
        print("7. SOM Cluster Quality Comparison")
        print("8. Exit")
        
        choice = input("Enter choice (1-8): ")
        
        if choice == '1':
            global_som = menu_1(X_scaled, X_raw)
        elif choice == '2':
            if global_som: menu_2(global_som, X_scaled)
            else: print("Please run Option 1 first.")
        elif choice == '3':
            menu_3(X_scaled)
        elif choice == '4':
            if global_som: menu_4(global_som, X_scaled, scaler, features)
            else: print("Please run Option 1 first.")
        elif choice == '5':
            if global_som: menu_5(global_som, scaler)
            else: print("Please run Option 1 first.")
        elif choice == '6':
            if global_som: menu_6(global_som, X_scaled, X_raw, customer_ids)
            else: print("Please run Option 1 first.")
        elif choice == '7':
            menu_7(X_scaled)
        elif choice == '8':
            print("Exiting System. Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()