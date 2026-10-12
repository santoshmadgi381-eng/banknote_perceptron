import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import math
import copy
import random

class StochasticFeatureSelector:
    def __init__(self, X_train, X_test, y_train, y_test, feature_names, lambda_penalty=0.02):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.lambda_penalty = lambda_penalty
        self.n_features = len(feature_names)
        self.clf = KNeighborsClassifier(n_neighbors=3)

    def calculate_energy(self, state):
        """Energy = Classification Error + Penalty for number of features"""
        if np.sum(state) == 0:
            return float('inf'), 0.0 # Invalid state
            
        features_idx = np.where(state == 1)[0]
        self.clf.fit(self.X_train[:, features_idx], self.y_train)
        preds = self.clf.predict(self.X_test[:, features_idx])
        acc = accuracy_score(self.y_test, preds)
        
        error = 1.0 - acc
        energy = error + self.lambda_penalty * np.sum(state)
        return energy, acc

    def generate_random_state(self):
        state = np.random.randint(0, 2, size=self.n_features)
        if np.sum(state) == 0:
            state[np.random.randint(0, self.n_features)] = 1
        return state

    def boltzmann_annealing(self, initial_state, initial_temp=10.0, max_iter=100, track_history=False):
        current_state = initial_state.copy()
        current_energy, current_acc = self.calculate_energy(current_state)
        
        best_state = current_state.copy()
        best_energy = current_energy
        best_acc = current_acc
        
        history = []
        
        for t in range(1, max_iter + 1):
            # Cooling schedule: T = T0 / ln(1 + t)
            temp = initial_temp / np.log(1 + t)
            
            # Perturbation: flip 1 random bit
            new_state = current_state.copy()
            flip_idx = np.random.randint(0, self.n_features)
            new_state[flip_idx] = 1 - new_state[flip_idx]
            
            if np.sum(new_state) == 0:
                continue
                
            new_energy, new_acc = self.calculate_energy(new_state)
            delta_e = new_energy - current_energy
            
            # Acceptance Probability
            if delta_e < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_e / temp)
                
            accepted = False
            if delta_e < 0 or random.random() < accept_prob:
                current_state = new_state
                current_energy = new_energy
                current_acc = new_acc
                accepted = True
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = current_state.copy()
                    best_acc = current_acc
            
            if track_history:
                history.append({
                    'iter': t, 'temp': temp, 'delta_e': delta_e, 
                    'prob': accept_prob, 'accepted': accepted,
                    'energy': current_energy, 'state': current_state.copy()
                })
                
        return best_state, best_energy, best_acc, history

    def cauchy_annealing(self, initial_state, initial_temp=10.0, max_iter=100, track_history=False):
        current_state = initial_state.copy()
        current_energy, current_acc = self.calculate_energy(current_state)
        
        best_state = current_state.copy()
        best_energy = current_energy
        best_acc = current_acc
        
        history = []
        
        for t in range(1, max_iter + 1):
            # Cooling schedule: T = T0 / (1 + t)
            temp = initial_temp / (1 + t)
            
            # Cauchy Perturbation mapping to discrete space
            # Use standard cauchy to determine number of bits to flip (longer jumps possible)
            cauchy_val = np.random.standard_cauchy() * temp
            num_flips = min(self.n_features, max(1, int(abs(cauchy_val))))
            
            new_state = current_state.copy()
            flip_indices = np.random.choice(self.n_features, num_flips, replace=False)
            for idx in flip_indices:
                new_state[idx] = 1 - new_state[idx]
                
            if np.sum(new_state) == 0:
                continue
                
            new_energy, new_acc = self.calculate_energy(new_state)
            delta_e = new_energy - current_energy
            
            # Acceptance Probability
            if delta_e < 0:
                accept_prob = 1.0
            else:
                accept_prob = temp / (temp**2 + delta_e**2) # Cauchy acceptance profile
                
            accepted = False
            if delta_e < 0 or random.random() < accept_prob:
                current_state = new_state
                current_energy = new_energy
                current_acc = new_acc
                accepted = True
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = current_state.copy()
                    best_acc = current_acc
            
            if track_history:
                history.append({
                    'iter': t, 'temp': temp, 'cauchy_val': cauchy_val, 'num_flips': num_flips,
                    'delta_e': delta_e, 'prob': accept_prob, 'accepted': accepted,
                    'energy': current_energy, 'state': current_state.copy()
                })
                
        return best_state, best_energy, best_acc, history

def load_and_prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    print(f"Loading dataset from: {url} ...")
    df = pd.read_csv(url, sep=';')
    
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    feature_names = df.columns[:-1].tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, feature_names

def print_features(state, feature_names):
    selected = [feature_names[i] for i in range(len(state)) if state[i] == 1]
    return selected

def analyze_history(history):
    energy_reductions = sum(1 for h in history if h['accepted'] and h['delta_e'] < 0)
    higher_accepted = sum(1 for h in history if h['accepted'] and h['delta_e'] > 0)
    
    states_str = [str(h['state']) for h in history]
    unique_states = len(set(states_str))
    repeated_states = len(states_str) - unique_states
    
    lowest_energy = min([h['energy'] for h in history])
    
    return energy_reductions, higher_accepted, repeated_states, unique_states, lowest_energy

def escape_challenge(selector, annealer_func, initial_state, name):
    current_state = initial_state.copy()
    best_energy, _ = selector.calculate_energy(current_state)
    
    trapped_iter = -1
    escape_iter = -1
    energy_after_escape = -1
    
    no_improve_count = 0
    t = 1
    
    while t <= 300: # Max limits for challenge
        temp = 10.0 / np.log(1+t) if name == "Boltzmann" else 10.0 / (1+t)
        
        new_state = current_state.copy()
        if name == "Boltzmann":
            new_state[np.random.randint(0, selector.n_features)] ^= 1
        else:
            flips = min(selector.n_features, max(1, int(abs(np.random.standard_cauchy() * temp))))
            for idx in np.random.choice(selector.n_features, flips, replace=False):
                new_state[idx] ^= 1
                
        if np.sum(new_state) == 0:
            t += 1
            continue
            
        new_energy, _ = selector.calculate_energy(new_state)
        delta_e = new_energy - best_energy
        
        # Local Minimum Detection (trapped for 10 iterations)
        if delta_e >= 0:
            no_improve_count += 1
            if no_improve_count == 10 and trapped_iter == -1:
                trapped_iter = t
        else:
            if trapped_iter != -1 and escape_iter == -1:
                escape_iter = t
                energy_after_escape = new_energy
                break # Escaped!
            best_energy = new_energy
            current_state = new_state
            no_improve_count = 0
            
        t += 1
        
    return trapped_iter, escape_iter, best_energy, energy_after_escape

def main():
    X_train, X_test, y_train, y_test, feature_names = load_and_prep_data()
    selector = StochasticFeatureSelector(X_train, X_test, y_train, y_test, feature_names)
    
    initial_state = None
    bm_history = None
    cm_history = None

    while True:
        print("\n" + "="*60)
        print(" BM & CM Stochastic Feature Selection - Wine Quality ")
        print("="*60)
        print("1. Generate Initial Feature State")
        print("2. Boltzmann Machine Feature Selection")
        print("3. Cauchy Machine Feature Selection")
        print("4. Energy State Transition Analyzer")
        print("5. Local Minimum Escape Challenge")
        print("6. Feature Stability Analyzer")
        print("7. Boltzmann vs Cauchy Optimization Battle")
        print("8. Best Feature Subset Finder")
        print("9. Stochastic Optimization Report")
        print("10. Exit")
        
        choice = input("Select an option (1-10): ")
        
        if choice == '1':
            initial_state = selector.generate_random_state()
            energy, acc = selector.calculate_energy(initial_state)
            print("\n--- 1. Initial Feature State ---")
            print(f"State Vector: {initial_state}")
            print(f"Active Features ({np.sum(initial_state)}): {print_features(initial_state, feature_names)}")
            print(f"Classification Accuracy: {acc*100:.2f}%")
            print(f"Initial Energy: {energy:.4f}")
            
        elif choice == '2':
            if initial_state is None: print("Run Option 1 first."); continue
            print("\n--- 2. Boltzmann Machine Feature Selection ---")
            b_state, b_energy, b_acc, bm_history = selector.boltzmann_annealing(initial_state, track_history=True)
            
            # Show last 5 transitions
            for h in bm_history[-5:]:
                print(f"Iter {h['iter']} | dE: {h['delta_e']:>7.4f} | Prob: {h['prob']:.4f} | Accepted: {h['accepted']}")
                
            print("\nFinal Boltzmann Result:")
            print(f"Best Energy: {b_energy:.4f} | Accuracy: {b_acc*100:.2f}%")
            print(f"Selected Features: {print_features(b_state, feature_names)}")
            
        elif choice == '3':
            if initial_state is None: print("Run Option 1 first."); continue
            print("\n--- 3. Cauchy Machine Feature Selection ---")
            c_state, c_energy, c_acc, cm_history = selector.cauchy_annealing(initial_state, track_history=True)
            
            # Show last 5 transitions
            for h in cm_history[-5:]:
                print(f"Iter {h['iter']} | CauchyPerturb: {h['cauchy_val']:>7.4f} (Flips: {h['num_flips']}) | dE: {h['delta_e']:>7.4f} | Accepted: {h['accepted']}")
                
            print("\nFinal Cauchy Result:")
            print(f"Minimum Energy Discovered: {c_energy:.4f} | Accuracy: {c_acc*100:.2f}%")
            print(f"Selected Features: {print_features(c_state, feature_names)}")
            
        elif choice == '4':
            if bm_history is None or cm_history is None: print("Run Options 2 and 3 first."); continue
            print("\n--- 4. Energy State Transition Analyzer ---")
            b_red, b_high, b_rep, b_uniq, b_min = analyze_history(bm_history)
            c_red, c_high, c_rep, c_uniq, c_min = analyze_history(cm_history)
            
            df = pd.DataFrame({
                "Metric": ["Energy Reductions", "Higher-Energy Accepted", "Repeated States", "Unique States Explored", "Lowest Energy"],
                "Boltzmann": [b_red, b_high, b_rep, b_uniq, round(b_min, 4)],
                "Cauchy": [c_red, c_high, c_rep, c_uniq, round(c_min, 4)]
            })
            print(df.to_string(index=False))
            
        elif choice == '5':
            if initial_state is None: print("Run Option 1 first."); continue
            print("\n--- 5. Local Minimum Escape Challenge ---")
            print("Detecting trap (10 iterations without improvement)...")
            
            b_trap, b_esc, b_trapped_E, b_esc_E = escape_challenge(selector, selector.boltzmann_annealing, initial_state, "Boltzmann")
            c_trap, c_esc, c_trapped_E, c_esc_E = escape_challenge(selector, selector.cauchy_annealing, initial_state, "Cauchy")
            
            def print_escape(name, trap, esc, e_trap, e_esc):
                if trap != -1 and esc != -1:
                    print(f"[{name}] Trapped at iter {trap} (E={e_trap:.4f}). Escaped at iter {esc} (New E={e_esc:.4f})")
                elif trap != -1:
                    print(f"[{name}] Trapped at iter {trap} but failed to escape within limit.")
                else:
                    print(f"[{name}] Never got trapped in a local minimum for 10 straight iterations.")
                    
            print_escape("Boltzmann", b_trap, b_esc, b_trapped_E, b_esc_E)
            print_escape("Cauchy", c_trap, c_esc, c_trapped_E, c_esc_E)
            
        elif choice == '6':
            print("\n--- 6. Feature Stability Analyzer ---")
            print("Executing BM and CM 20 times each (Total 40 runs)...")
            bm_counts = np.zeros(selector.n_features)
            cm_counts = np.zeros(selector.n_features)
            
            for _ in range(20):
                start = selector.generate_random_state()
                bs, _, _, _ = selector.boltzmann_annealing(start, max_iter=50)
                cs, _, _, _ = selector.cauchy_annealing(start, max_iter=50)
                bm_counts += bs
                cm_counts += cs
                
            total_counts = bm_counts + cm_counts
            stability = []
            for i, count in enumerate(total_counts):
                freq = count / 40.0
                if freq >= 0.7: status = "Highly Stable"
                elif freq >= 0.4: status = "Moderately Stable"
                else: status = "Rarely Selected"
                stability.append((feature_names[i], freq*100, status))
                
            stability.sort(key=lambda x: x[1], reverse=True)
            for rank, (feat, freq, stat) in enumerate(stability, 1):
                print(f"{rank}. {feat:<20} | Selection Freq: {freq:>5.1f}% | {stat}")
                
        elif choice == '7':
            if initial_state is None: print("Run Option 1 first."); continue
            print("\n--- 7. Boltzmann vs Cauchy Optimization Battle ---")
            b_state, b_energy, b_acc, bh = selector.boltzmann_annealing(initial_state, max_iter=150, track_history=True)
            c_state, c_energy, c_acc, ch = selector.cauchy_annealing(initial_state, max_iter=150, track_history=True)
            
            b_metrics = analyze_history(bh)
            c_metrics = analyze_history(ch)
            
            df = pd.DataFrame({
                "Metric": ["Min Energy", "Final Accuracy", "Unique States Explored", "Higher-Energy Accepted"],
                "Boltzmann": [f"{b_energy:.4f}", f"{b_acc*100:.2f}%", b_metrics[3], b_metrics[1]],
                "Cauchy": [f"{c_energy:.4f}", f"{c_acc*100:.2f}%", c_metrics[3], c_metrics[1]]
            })
            print(df.to_string(index=False))
            
            winner = "Boltzmann" if b_energy < c_energy else "Cauchy" if c_energy < b_energy else "Tie"
            print(f"\nWinner based on Minimum Energy: {winner}")
            
        elif choice == '8':
            print("\n--- 8. Best Feature Subset Finder ---")
            print("Running multi-start search (5 random initializations)...")
            best_overall_state = None
            best_overall_energy = float('inf')
            best_overall_acc = 0
            
            for i in range(5):
                start = selector.generate_random_state()
                bs, be, bacc, _ = selector.boltzmann_annealing(start, max_iter=100)
                cs, ce, cacc, _ = selector.cauchy_annealing(start, max_iter=100)
                
                if be < best_overall_energy:
                    best_overall_energy, best_overall_state, best_overall_acc = be, bs, bacc
                if ce < best_overall_energy:
                    best_overall_energy, best_overall_state, best_overall_acc = ce, cs, cacc
                    
            selected = print_features(best_overall_state, feature_names)
            removed = [f for f in feature_names if f not in selected]
            
            print(f"\nOverall Best Energy Discovered: {best_overall_energy:.4f}")
            print(f"Classification Accuracy: {best_overall_acc*100:.2f}%")
            print(f"Selected Features ({len(selected)}): {selected}")
            print(f"Removed Features ({len(removed)}): {removed}")
            
        elif choice == '9':
            if initial_state is None: print("Run Option 1 first."); continue
            print("\n--- 9. Stochastic Optimization Report ---")
            # Generating report dynamically based on current initial state
            bs, be, bacc, bh = selector.boltzmann_annealing(initial_state, track_history=True)
            cs, ce, cacc, ch = selector.cauchy_annealing(initial_state, track_history=True)
            
            b_red, b_high, b_rep, b_uniq, b_min = analyze_history(bh)
            c_red, c_high, c_rep, c_uniq, c_min = analyze_history(ch)
            
            init_energy, _ = selector.calculate_energy(initial_state)
            
            report = pd.DataFrame({
                "Machine Type": ["Boltzmann Machine", "Cauchy Machine"],
                "Initial Energy": [round(init_energy, 4)] * 2,
                "Final Energy": [round(be, 4), round(ce, 4)],
                "Selected Features": [np.sum(bs), np.sum(cs)],
                "Accuracy": [f"{bacc*100:.2f}%", f"{cacc*100:.2f}%"],
                "Unique States": [b_uniq, c_uniq],
                "Search Iterations": [100, 100]
            })
            print(report.to_string(index=False))
            print("\nConclusion: Cauchy machines generally explore state space faster due to longer jump capabilities (more flipped bits), whereas Boltzmann machines perform finer local exploitation.")
            
        elif choice == '10':
            print("Exiting Stochastic Feature Selection System. Goodbye!")
            break
        else:
            print("Invalid input. Please choose between 1 and 10.")

if __name__ == "__main__":
    main()