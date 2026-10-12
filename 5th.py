import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import random
import copy

class BAM:
    def __init__(self):
        self.weights = None
        self.weights_T = None

    def fit(self, X, Y):
        """Store associations using the correlation learning rule."""
        # X: (P, N) bipolar matrix, Y: (P, M) bipolar matrix
        # W = sum(X_i^T * Y_i) -> shape (N, M)
        self.weights = np.dot(X.T, Y)
        self.weights_T = self.weights.T
        return self

    def activation(self, net_input, previous_state=None):
        """Bipolar activation. If net is 0, keeps previous state."""
        out = np.where(net_input > 0, 1, -1)
        if previous_state is not None:
            out = np.where(net_input == 0, previous_state, out)
        return out

    def forward(self, x):
        """Forward Recall: X -> Y"""
        net_y = np.dot(x, self.weights)
        return self.activation(net_y)

    def backward(self, y):
        """Backward Recall: Y -> X"""
        net_x = np.dot(y, self.weights_T)
        return self.activation(net_x)
        
    def bidirectional_recall(self, x_init, max_iter=10):
        """Repeated forward and backward recall until stable."""
        x_curr = x_init.copy()
        y_curr = self.forward(x_curr)
        
        for _ in range(max_iter):
            x_next = self.backward(y_curr)
            y_next = self.forward(x_next)
            
            if np.array_equal(x_curr, x_next) and np.array_equal(y_curr, y_next):
                break
                
            x_curr = x_next
            y_curr = y_next
            
        return x_curr, y_curr

def load_and_prep_data():
    """Loads digits, converts to bipolar (-1, 1), and creates bipolar targets."""
    digits = load_digits()
    
    # Select one representative sample for each digit 0-9
    X_reps = []
    y_labels = []
    
    for i in range(10):
        # Find first occurrence of digit i
        idx = np.where(digits.target == i)[0][0]
        # Binarize: values > 8 become 1, else -1
        bipolar_img = np.where(digits.data[idx] > 8, 1, -1)
        X_reps.append(bipolar_img)
        y_labels.append(i)
        
    X_reps = np.array(X_reps)
    
    # Create unique bipolar output codes (One-Hot mapped to bipolar)
    # Shape: (10, 10), diagonal is 1, rest is -1
    Y_codes = np.full((10, 10), -1)
    np.fill_diagonal(Y_codes, 1)
    
    return X_reps, Y_codes, y_labels

def print_digit(bipolar_array):
    """Utility to print 8x8 bipolar digit."""
    arr = bipolar_array.reshape(8, 8)
    for row in arr:
        print("".join(["██" if val == 1 else "  " for val in row]))
    print("-" * 16)

def identify_digit(code, Y_codes):
    """Finds which stored code matches the recalled code."""
    for i, c in enumerate(Y_codes):
        if np.array_equal(code, c):
            return i
    return -1 # Unrecognized/Spurious state

def menu_1(X, Y):
    print("\n--- 1. Store Digit-Code Pattern Associations ---")
    bam = BAM()
    bam.fit(X, Y)
    print(f"Network trained on {len(X)} digit-code pairs.")
    print(f"Forward Weight Matrix (W) Shape: {bam.weights.shape}")
    print(f"Transpose Matrix (W^T) Shape: {bam.weights_T.shape}")
    
    print("\nWeight Matrix Sample (First 5 rows, 10 columns):")
    print(bam.weights[:5, :])
    return bam

def menu_2(bam, X, Y, labels):
    print("\n--- 2. Forward Digit Recall Challenge ---")
    idx = int(input("Enter digit to recall (0-9): "))
    
    x_input = X[idx]
    y_recalled = bam.forward(x_input)
    identified = identify_digit(y_recalled, Y)
    
    print("\nInput Pattern (X):")
    print_digit(x_input)
    print(f"Calculated Y-Layer State: {y_recalled}")
    print(f"Identified Digit: {identified if identified != -1 else 'Unrecognized/Spurious'}")

def menu_3(bam, X, Y, labels):
    print("\n--- 3. Reverse Digit Reconstruction Challenge ---")
    idx = int(input("Enter digit code to reconstruct (0-9): "))
    
    y_input = Y[idx]
    x_reconstructed = bam.backward(y_input)
    
    print(f"\nInput Code (Y): {y_input}")
    print("\nReconstructed Pattern (X_reconstructed):")
    print_digit(x_reconstructed)
    
    print("Original Stored Pattern (X_original):")
    print_digit(X[idx])
    
    match = np.array_equal(x_reconstructed, X[idx])
    print(f"Perfect Reconstruction: {match}")

def menu_4(bam, X, Y):
    print("\n--- 4. Noisy Pattern Association Test ---")
    idx = int(input("Select a digit to add noise to (0-9): "))
    x_original = X[idx]
    
    noise_levels = [0.05, 0.10, 0.20, 0.30, 0.40]
    max_tolerated = 0
    
    for noise in noise_levels:
        x_noisy = x_original.copy()
        # Flip random pixels
        num_flips = int(len(x_noisy) * noise)
        flip_indices = np.random.choice(len(x_noisy), num_flips, replace=False)
        x_noisy[flip_indices] *= -1
        
        y_recalled = bam.forward(x_noisy)
        identified = identify_digit(y_recalled, Y)
        
        correct = (identified == idx)
        if correct:
            max_tolerated = noise * 100
            
        print(f"\nNoise: {noise*100}% (Pixels flipped: {num_flips})")
        print(f"Recalled Digit: {identified}, Correct: {correct}")
        
    print(f"\nMaximum Noise Level Tolerated: {max_tolerated}%")

def menu_5(bam, X, Y):
    print("\n--- 5. Partial Digit Completion Challenge ---")
    idx = int(input("Select digit to partially erase (0-9): "))
    x_original = X[idx]
    x_erased = x_original.copy().reshape(8, 8)
    
    # Erase bottom half (set to -1)
    x_erased[4:, :] = -1
    x_erased = x_erased.flatten()
    
    print("Incomplete Input Pattern (Bottom half erased):")
    print_digit(x_erased)
    
    x_stable, y_stable = bam.bidirectional_recall(x_erased)
    
    print("Reconstructed Pattern after Bidirectional Iterations:")
    print_digit(x_stable)
    
    correct_pixels = np.sum(x_stable == x_original)
    accuracy = (correct_pixels / len(x_original)) * 100
    identified = identify_digit(y_stable, Y)
    
    print(f"Identified Digit: {identified}")
    print(f"Pixel Reconstruction Accuracy: {accuracy:.2f}%")

def menu_6(bam, X, Y):
    print("\n--- 6. Bidirectional Recall Consistency Analyzer ---")
    failed = []
    
    for idx in range(len(X)):
        x_input = X[idx]
        # X -> Y
        y_out = bam.forward(x_input)
        # Y -> X
        x_reconstructed = bam.backward(y_out)
        
        consistency = np.sum(x_input == x_reconstructed) / len(x_input) * 100
        
        print(f"Digit {idx} -> Consistency: {consistency:.2f}%")
        
        if not np.array_equal(x_input, x_reconstructed):
            failed.append(idx)
            
    if failed:
        print(f"\nAssociations failing to return original pattern: {failed}")
    else:
        print("\nAll stored associations are 100% bidirectionally consistent!")

def menu_7(X, Y):
    print("\n--- 7. Association Capacity Challenge ---")
    configs = [2, 4, 6, 8, 10]
    
    for capacity in configs:
        b = BAM().fit(X[:capacity], Y[:capacity])
        
        fw_correct = 0
        bw_correct = 0
        consistent = 0
        
        for i in range(capacity):
            y_rec = b.forward(X[i])
            x_rec = b.backward(Y[i])
            
            if identify_digit(y_rec, Y) == i:
                fw_correct += 1
            if np.array_equal(x_rec, X[i]):
                bw_correct += 1
            if np.array_equal(b.backward(b.forward(X[i])), X[i]):
                consistent += 1
                
        print(f"\nCapacity: {capacity} Pairs Stored")
        print(f"Forward Accuracy: {(fw_correct/capacity)*100:.2f}%")
        print(f"Backward Accuracy: {(bw_correct/capacity)*100:.2f}%")
        print(f"Bidirectional Consistency: {(consistent/capacity)*100:.2f}%")

def menu_8(bam, X, Y):
    print("\n--- 8. Cross-Association Error Detector ---")
    errors = []
    
    for idx in range(len(X)):
        y_out = bam.forward(X[idx])
        identified = identify_digit(y_out, Y)
        
        if identified != idx:
            errors.append((idx, identified))
            print(f"\nInput Digit: {idx}")
            print(f"Expected Code Index: {idx}")
            print(f"Recalled Code Index: {identified}")
            print(f"Incorrectly Associated With: {identified if identified != -1 else 'Spurious State'}")
            
    if not errors:
        print("No cross-association errors detected. BAM has safely stored all patterns.")
    else:
        print(f"\nTotal Errors: {len(errors)}")

def menu_9(X, Y):
    print("\n--- 9. BAM Associative Memory Report ---")
    bam = BAM().fit(X, Y)
    
    fw_acc = sum([identify_digit(bam.forward(X[i]), Y) == i for i in range(len(X))]) / len(X)
    bw_acc = sum([np.array_equal(bam.backward(Y[i]), X[i]) for i in range(len(X))]) / len(X)
    
    print(f"Stored Associations: {len(X)}")
    print(f"Overall Forward Recall Accuracy: {fw_acc*100:.2f}%")
    print(f"Overall Backward Recall Accuracy: {bw_acc*100:.2f}%")
    print("Maximum Reliable Capacity (based on Option 7): Varies by pattern orthogonality (usually decreases past 4-6 patterns in dense vectors).")
    
def main():
    X, Y, labels = load_and_prep_data()
    global_bam = None

    while True:
        print("\n" + "="*55)
        print("   BAM Handwritten Digit Association System   ")
        print("="*55)
        print("1. Store Digit-Code Pattern Associations")
        print("2. Forward Digit Recall Challenge")
        print("3. Reverse Digit Reconstruction Challenge")
        print("4. Noisy Pattern Association Test")
        print("5. Partial Digit Completion Challenge")
        print("6. Bidirectional Recall Consistency Analyzer")
        print("7. Association Capacity Challenge")
        print("8. Cross-Association Error Detector")
        print("9. BAM Associative Memory Report")
        print("10. Exit")
        
        choice = input("Enter choice (1-10): ")
        
        if choice == '1':
            global_bam = menu_1(X, Y)
        elif choice == '2':
            if global_bam: menu_2(global_bam, X, Y, labels)
            else: print("Please run Option 1 first.")
        elif choice == '3':
            if global_bam: menu_3(global_bam, X, Y, labels)
            else: print("Please run Option 1 first.")
        elif choice == '4':
            if global_bam: menu_4(global_bam, X, Y)
            else: print("Please run Option 1 first.")
        elif choice == '5':
            if global_bam: menu_5(global_bam, X, Y)
            else: print("Please run Option 1 first.")
        elif choice == '6':
            if global_bam: menu_6(global_bam, X, Y)
            else: print("Please run Option 1 first.")
        elif choice == '7':
            menu_7(X, Y)
        elif choice == '8':
            if global_bam: menu_8(global_bam, X, Y)
            else: print("Please run Option 1 first.")
        elif choice == '9':
            menu_9(X, Y)
        elif choice == '10':
            print("Exiting BAM System. Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()