import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import random

class HopfieldNetwork:
    def __init__(self, size=64):
        self.size = size
        self.weights = np.zeros((size, size))
        
    def fit(self, patterns):
        """Train the Hopfield Network using the Hebbian learning rule."""
        self.weights = np.zeros((self.size, self.size))
        for p in patterns:
            self.weights += np.outer(p, p)
            
        # No self-connections
        np.fill_diagonal(self.weights, 0)
        # Normalize weights
        self.weights /= len(patterns)
        return self
        
    def recall(self, pattern, max_iter=100):
        """Synchronous update until stable state is reached."""
        state = pattern.copy()
        for iteration in range(max_iter):
            # Compute net input and apply sign activation function
            net_input = np.dot(self.weights, state)
            new_state = np.where(net_input >= 0, 1, -1)
            
            # Check for convergence
            if np.array_equal(state, new_state):
                return new_state, iteration + 1
            state = new_state.copy()
            
        return state, max_iter

def load_and_prep_data():
    """
    Dataset: Optical Recognition of Handwritten Digits
    URL: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
    """
    print("Loading Optical Recognition of Handwritten Digits Dataset...")
    digits = load_digits()
    
    # Extract one representative sample per digit (0-9)
    patterns = []
    labels = []
    
    for i in range(10):
        idx = np.where(digits.target == i)[0][0]
        # Bipolar conversion: values > 8 are foreground (+1), else background (-1)
        bipolar = np.where(digits.data[idx] > 8, 1, -1)
        patterns.append(bipolar)
        labels.append(i)
        
    return np.array(patterns), np.array(labels)

def print_digit(pattern):
    """Prints a 64-element bipolar array as an 8x8 ASCII image."""
    arr = pattern.reshape(8, 8)
    for row in arr:
        print("".join(["██" if val == 1 else "  " for val in row]))
    print("-" * 16)

def identify_pattern(recalled_pattern, stored_patterns):
    """Finds the matching stored pattern index. Returns -1 if it's a spurious state."""
    for i, p in enumerate(stored_patterns):
        if np.array_equal(recalled_pattern, p):
            return i
    # Check inverse states (Hopfield networks also store inverses)
    for i, p in enumerate(stored_patterns):
        if np.array_equal(recalled_pattern, -p):
            return -2 # Indicates an inverse spurious state
    return -1

def menu_1(patterns, labels):
    print("\n--- 1. Store Handwritten Digit Patterns ---")
    # By default, Hopfield capacity is ~0.138 * N. For N=64, reliable capacity is ~9 patterns.
    # We will start by storing the first 5 digits (0, 1, 2, 3, 4) for stability.
    num_to_store = 5
    stored_patterns = patterns[:num_to_store]
    
    hnn = HopfieldNetwork(size=64)
    hnn.fit(stored_patterns)
    
    print(f"Hopfield Network initialized and trained on Digits {list(labels[:num_to_store])}.")
    print(f"Weight Matrix Shape: {hnn.weights.shape}")
    print("\nSample Stored Pattern (Digit 0):")
    print_digit(stored_patterns[0])
    
    return hnn, stored_patterns

def menu_2(hnn, stored_patterns):
    print("\n--- 2. Recall a Noisy Digit Pattern ---")
    idx = int(input(f"Select a digit index to corrupt (0 to {len(stored_patterns)-1}): "))
    noise_level = float(input("Enter corruption percentage (e.g., 10, 20, 30): ")) / 100.0
    
    original = stored_patterns[idx]
    noisy = original.copy()
    
    # Flip random pixels
    num_flips = int(64 * noise_level)
    flip_indices = np.random.choice(64, num_flips, replace=False)
    noisy[flip_indices] *= -1
    
    recalled, iters = hnn.recall(noisy)
    
    print("\nOriginal Digit:")
    print_digit(original)
    print(f"Corrupted Digit ({int(noise_level*100)}% noise):")
    print_digit(noisy)
    print(f"Recalled Digit (Converged in {iters} iterations):")
    print_digit(recalled)
    
    match = identify_pattern(recalled, stored_patterns)
    print(f"Match Status: {'Success! Matched Digit ' + str(match) if match == idx else 'Failed.'}")

def menu_3(hnn, stored_patterns):
    print("\n--- 3. Missing Pixel Reconstruction Challenge ---")
    idx = int(input(f"Select a digit to reconstruct (0 to {len(stored_patterns)-1}): "))
    original = stored_patterns[idx]
    incomplete = original.copy().reshape(8, 8)
    
    print("Select corruption type: 1) Row 2) Column 3) Block")
    choice = input("Choice: ")
    
    if choice == '1':
        incomplete[4, :] = -1
        print("Erased Row 4.")
    elif choice == '2':
        incomplete[:, 4] = -1
        print("Erased Column 4.")
    else:
        incomplete[4:8, 4:8] = -1
        print("Erased Bottom-Right Block (4x4).")
        
    incomplete = incomplete.flatten()
    recalled, _ = hnn.recall(incomplete)
    
    print("\nIncomplete Pattern:")
    print_digit(incomplete)
    print("Reconstructed Pattern:")
    print_digit(recalled)
    
    acc = np.sum(recalled == original) / 64 * 100
    print(f"Reconstruction Accuracy: {acc:.2f}%")

def menu_4(hnn, stored_patterns):
    print("\n--- 4. Digit Corruption Tolerance Test ---")
    noise_levels = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    for i, original in enumerate(stored_patterns):
        max_tolerated = 0
        for noise in noise_levels:
            noisy = original.copy()
            flips = int(64 * noise)
            idx_flips = np.random.choice(64, flips, replace=False)
            noisy[idx_flips] *= -1
            
            recalled, _ = hnn.recall(noisy)
            if np.array_equal(recalled, original):
                max_tolerated = noise * 100
            else:
                break
        print(f"Digit {i} -> Maximum Noise Tolerated: {max_tolerated}%")

def menu_5(all_patterns):
    print("\n--- 5. Hopfield Memory Capacity Challenge ---")
    capacities = [2, 3, 4, 5, 6, 8, 10]
    
    for cap in capacities:
        patterns = all_patterns[:cap]
        h = HopfieldNetwork(64).fit(patterns)
        
        success = 0
        for i, original in enumerate(patterns):
            # Test with 10% noise
            noisy = original.copy()
            flips = int(64 * 0.1)
            idx_flips = np.random.choice(64, flips, replace=False)
            noisy[idx_flips] *= -1
            
            recalled, _ = h.recall(noisy)
            if np.array_equal(recalled, original):
                success += 1
                
        print(f"Stored Patterns: {cap} | Successful Recalls: {success} | Failed: {cap - success} | Accuracy: {(success/cap)*100:.1f}%")

def menu_6(hnn, stored_patterns):
    print("\n--- 6. False Memory Detector ---")
    # Generate a completely random bipolar pattern
    random_pattern = np.random.choice([-1, 1], size=64)
    
    recalled, iters = hnn.recall(random_pattern)
    match_idx = identify_pattern(recalled, stored_patterns)
    
    print("\nRandom Input Pattern:")
    print_digit(random_pattern)
    print(f"Converged State (after {iters} iterations):")
    print_digit(recalled)
    
    if match_idx >= 0:
        print(f"Result: Matched Stored Digit {match_idx}")
    elif match_idx == -2:
        print("Result: Spurious State (Inverse False Memory)")
    else:
        print("Result: Spurious State (False Memory detected!)")

def menu_7(hnn, stored_patterns):
    print("\n--- 7. Digit Confusion Analyzer ---")
    confusions = {}
    
    # Test each digit with high noise to force misclassification
    for i, original in enumerate(stored_patterns):
        noisy = original.copy()
        # 35% noise is usually enough to flip Hopfield basins for 8x8 dense vectors
        flips = int(64 * 0.35) 
        idx_flips = np.random.choice(64, flips, replace=False)
        noisy[idx_flips] *= -1
        
        recalled, _ = hnn.recall(noisy)
        match_idx = identify_pattern(recalled, stored_patterns)
        
        if match_idx >= 0 and match_idx != i:
            key = f"Digit {i} -> Recalled as Digit {match_idx}"
            confusions[key] = confusions.get(key, 0) + 1
            print(key)
            
    if not confusions:
        print("No confusions detected. Digits mostly fall into spurious states when heavily corrupted.")
    else:
        print("\nMost frequent confusions at 35% noise:")
        for k, v in confusions.items():
            print(f"{k} (Count: {v})")

def menu_8(hnn, stored_patterns):
    print("\n--- 8. Hopfield Recall Performance Report ---")
    report = []
    
    for i, original in enumerate(stored_patterns):
        noisy = original.copy()
        noise_pct = 0.15
        flips = int(64 * noise_pct)
        idx_flips = np.random.choice(64, flips, replace=False)
        noisy[idx_flips] *= -1
        
        recalled, iters = hnn.recall(noisy)
        match_idx = identify_pattern(recalled, stored_patterns)
        
        status = "Correct" if match_idx == i else "Spurious" if match_idx < 0 else f"Confused with {match_idx}"
        acc = np.sum(recalled == original) / 64 * 100
        
        report.append({
            "Target Digit": i,
            "Corruption %": int(noise_pct * 100),
            "Flipped Pixels": flips,
            "Iterations": iters,
            "Reconstruction %": f"{acc:.1f}%",
            "Recalled Digit": match_idx if match_idx >= 0 else "N/A",
            "Status": status
        })
        
    df = pd.DataFrame(report)
    print(df.to_string(index=False))

def main():
    try:
        all_patterns, labels = load_and_prep_data()
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return

    global_hnn = None
    stored_patterns = None

    while True:
        print("\n" + "="*55)
        print("   Hopfield Network Associative Memory System   ")
        print("="*55)
        print("1. Store Handwritten Digit Patterns")
        print("2. Recall a Noisy Digit Pattern")
        print("3. Missing Pixel Reconstruction Challenge")
        print("4. Digit Corruption Tolerance Test")
        print("5. Hopfield Memory Capacity Challenge")
        print("6. False Memory Detector")
        print("7. Digit Confusion Analyzer")
        print("8. Hopfield Recall Performance Report")
        print("9. Exit")
        
        choice = input("Enter choice (1-9): ")
        
        if choice == '1':
            global_hnn, stored_patterns = menu_1(all_patterns, labels)
        elif choice == '2':
            if global_hnn: menu_2(global_hnn, stored_patterns)
            else: print("Please run Option 1 first.")
        elif choice == '3':
            if global_hnn: menu_3(global_hnn, stored_patterns)
            else: print("Please run Option 1 first.")
        elif choice == '4':
            if global_hnn: menu_4(global_hnn, stored_patterns)
            else: print("Please run Option 1 first.")
        elif choice == '5':
            menu_5(all_patterns)
        elif choice == '6':
            if global_hnn: menu_6(global_hnn, stored_patterns)
            else: print("Please run Option 1 first.")
        elif choice == '7':
            if global_hnn: menu_7(global_hnn, stored_patterns)
            else: print("Please run Option 1 first.")
        elif choice == '8':
            if global_hnn: menu_8(global_hnn, stored_patterns)
            else: print("Please run Option 1 first.")
        elif choice == '9':
            print("Exiting Hopfield System. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 9.")

if __name__ == "__main__":
    main()