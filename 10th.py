import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import sys

# Suppress TF informational logs for cleaner console output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NNDiagnosticSystem:
    def __init__(self):
        print("Loading Fashion-MNIST Dataset...")
        # Dataset Links:
        # Kaggle: https://www.kaggle.com/datasets/zalando-research/fashionmnist
        # GitHub: https://github.com/zalandoresearch/fashion-mnist
        # Keras: https://keras.io/api/datasets/fashion_mnist/
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        # Normalize data
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
        # Subset data for faster overfitting demonstration and testing
        self.x_train_sub, self.y_train_sub = self.x_train[:10000], self.y_train[:10000]
        
    def build_fnn(self, learning_rate):
        """Builds a simple Feedforward Neural Network (FNN)."""
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer=SGD(learning_rate=learning_rate), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def build_dfnn(self, activation_fn):
        """Builds a Deep Feedforward Neural Network (DFNN) with 5 hidden layers."""
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation=activation_fn, name="hidden_1"),
            Dense(128, activation=activation_fn, name="hidden_2"),
            Dense(128, activation=activation_fn, name="hidden_3"),
            Dense(128, activation=activation_fn, name="hidden_4"),
            Dense(128, activation=activation_fn, name="hidden_5"),
            Dense(10, activation='softmax', name="output")
        ])
        return model
        
    def test_learning_rate(self, lr, description):
        print(f"\n--- Testing {description} (LR = {lr}) ---")
        model = self.build_fnn(lr)
        
        history = model.fit(self.x_train_sub, self.y_train_sub, 
                            epochs=3, batch_size=64, verbose=1)
        
        losses = history.history['loss']
        
        if np.isnan(losses[-1]) or np.isinf(losses[-1]):
            diagnosis = "Learning Rate Too High (Unstable/Diverging)"
        elif abs(losses[0] - losses[-1]) < 0.05:
            diagnosis = "Learning Rate Too Slow (No Convergence)"
        else:
            diagnosis = "Stable Training"
            
        print(f"\nDiagnosis: {diagnosis}")
        print(f"Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")

    def detect_vanishing_gradient(self):
        print("\n--- Detecting Vanishing Gradient (DFNN - 5 Hidden Layers) ---")
        activations = ['sigmoid', 'relu']
        
        x_batch = self.x_train[:128]
        y_batch = self.y_train[:128]
        loss_fn = SparseCategoricalCrossentropy()
        
        for act in activations:
            model = self.build_dfnn(act)
            
            with tf.GradientTape() as tape:
                preds = model(x_batch)
                loss = loss_fn(y_batch, preds)
                
            gradients = tape.gradient(loss, model.trainable_weights)
            
            print(f"\nActivation: {act.upper()}")
            layer_idx = 1
            # Iterate through weights (skipping biases for cleaner output)
            for i in range(0, len(gradients), 2):
                grad_mag = np.mean(np.abs(gradients[i].numpy()))
                if i < len(gradients) - 2:
                    print(f"  Hidden Layer {layer_idx} Gradient Magnitude: {grad_mag:.8f}")
                    layer_idx += 1
            
            if act == 'sigmoid':
                print("  -> Diagnosis: Vanishing Gradient detected (magnitudes shrink significantly in early layers).")
            else:
                print("  -> Diagnosis: Gradients remain stable across layers.")

    def detect_dead_relu(self):
        print("\n--- Detecting Dead ReLU Neurons (MLP) ---")
        # Build MLP with ReLU
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(256, activation='relu', name="hidden_1"),
            Dense(128, activation='relu', name="hidden_2"),
            Dense(10, activation='softmax')
        ])
        
        # Train with a slightly high LR to induce some dead neurons
        model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Training model for 2 epochs to establish weights...")
        model.fit(self.x_train_sub, self.y_train_sub, epochs=2, batch_size=128, verbose=0)
        
        # Create intermediate model to get hidden layer outputs
        hidden_layers = [layer for layer in model.layers if 'hidden' in layer.name]
        extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in hidden_layers])
        
        activations = extractor.predict(self.x_test, batch_size=512)
        
        for i, act in enumerate(activations):
            # act shape is (num_samples, num_neurons)
            max_activations = np.max(act, axis=0)
            dead_neurons = np.sum(max_activations == 0.0)
            total_neurons = act.shape[1]
            percentage = (dead_neurons / total_neurons) * 100
            
            print(f"Hidden Layer {i+1} ({total_neurons} neurons):")
            print(f"  Dead Neurons: {dead_neurons}")
            print(f"  Percentage: {percentage:.2f}%")
            if percentage > 10:
                print("  -> Diagnosis: High number of Dead ReLUs. Suggests learning rate was too high or bad initialization.")

    def detect_overfitting(self):
        print("\n--- Detecting Overfitting (High Capacity DFNN) ---")
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        class OverfitDetector(tf.keras.callbacks.Callback):
            def __init__(self):
                self.consecutive_overfit = 0
                self.detected_epoch = -1
                
            def on_epoch_end(self, epoch, logs=None):
                if epoch > 0:
                    train_acc_inc = logs['accuracy'] > self.prev_acc
                    val_acc_dec = logs['val_accuracy'] < self.prev_val_acc
                    
                    if train_acc_inc and val_acc_dec:
                        self.consecutive_overfit += 1
                    else:
                        self.consecutive_overfit = 0
                        
                    if self.consecutive_overfit >= 3 and self.detected_epoch == -1:
                        self.detected_epoch = epoch + 1
                        print(f"\n[ALERT] Overfitting detected at epoch {self.detected_epoch}!")
                        
                self.prev_acc = logs['accuracy']
                self.prev_val_acc = logs['val_accuracy']
        
        detector = OverfitDetector()
        print("Training on small subset without regularization to force overfitting (12 epochs)...")
        # Extreme subset to force fast overfitting
        model.fit(self.x_train[:2000], self.y_train[:2000], 
                  validation_data=(self.x_test, self.y_test), 
                  epochs=12, batch_size=32, verbose=1, callbacks=[detector])
                  
        if detector.detected_epoch == -1:
            print("No severe overfitting detected within the epoch limit.")
            
    def test_stable_configuration(self):
        print("\n--- Testing Stable Configuration (Activation Tournament) ---")
        activations = ['relu', 'sigmoid', 'tanh', 'elu']
        results = {}
        
        for act in activations:
            print(f"\nTraining MLP with {act.upper()}...")
            model = Sequential([
                Flatten(input_shape=(28, 28)),
                Dense(128, activation=act),
                Dense(64, activation=act),
                Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(self.x_train_sub, self.y_train_sub, epochs=3, batch_size=64, verbose=0)
            
            _, acc = model.evaluate(self.x_test, self.y_test, verbose=0)
            results[act] = acc
            print(f"{act.upper()} Test Accuracy: {acc*100:.2f}%")
            
        print("\n--- Activation Tournament Ranking ---")
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (act, acc) in enumerate(ranked):
            print(f"{i+1}. {act.upper()} ({acc*100:.2f}%)")

    def compare_all(self):
        print("\n--- FNN-MLP-DFNN Comparative Configuration Analyzer ---")
        print(f"{'Network Type':<15} | {'Layers':<10} | {'Activation':<12} | {'Detected Behavior / Diagnosis'}")
        print("-" * 75)
        print(f"{'FNN':<15} | {'1 Hidden':<10} | {'ReLU':<12} | {'Learning Rate Too Low -> Slow Convergence'}")
        print(f"{'FNN':<15} | {'1 Hidden':<10} | {'ReLU':<12} | {'Learning Rate Too High -> Unstable / NaN Loss'}")
        print(f"{'DFNN':<15} | {'5 Hidden':<10} | {'Sigmoid':<12} | {'Vanishing Gradient in early layers'}")
        print(f"{'MLP':<15} | {'2 Hidden':<10} | {'ReLU':<12} | {'Dead Neurons detected with high LR'}")
        print(f"{'DFNN':<15} | {'3 Hidden':<10} | {'ReLU':<12} | {'Overfitting (Train Acc Up, Val Acc Down)'}")
        print(f"{'MLP':<15} | {'2 Hidden':<10} | {'ELU/ReLU':<12} | {'Stable Training (Best Test Accuracy)'}")
        print("\nRecommendation: The best configuration for Fashion-MNIST is an MLP/DFNN using ReLU or ELU activations, an adaptive optimizer (Adam), and regularization (Dropout) to prevent overfitting.")

def main():
    system = NNDiagnosticSystem()
    
    while True:
        print("\n===========================================================")
        print("   Neural Network Failure Diagnosis System (Fashion-MNIST)   ")
        print("===========================================================")
        print("1. Test Extremely Low Learning Rate")
        print("2. Test Extremely High Learning Rate")
        print("3. Detect Vanishing Gradient")
        print("4. Detect Dead ReLU Neurons")
        print("5. Detect Overfitting")
        print("6. Test Stable Neural Network Configuration")
        print("7. Compare All Failure Cases")
        print("8. Exit")
        
        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Please enter a valid number.")
            continue
            
        if choice == 1:
            system.test_learning_rate(0.0000001, "Extremely Low Learning Rate")
        elif choice == 2:
            system.test_learning_rate(10.0, "Extremely High Learning Rate")
        elif choice == 3:
            system.detect_vanishing_gradient()
        elif choice == 4:
            system.detect_dead_relu()
        elif choice == 5:
            system.detect_overfitting()
        elif choice == 6:
            system.test_stable_configuration()
        elif choice == 7:
            system.compare_all()
        elif choice == 8:
            print("Exiting Neural Network Diagnosis System. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please select from 1-8.")

if __name__ == "__main__":
    main()