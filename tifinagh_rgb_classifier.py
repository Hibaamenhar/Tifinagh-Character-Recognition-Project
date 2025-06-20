import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_data(data_path, image_size=(32, 32)):
    X, y = [], []
    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    if not classes:
        raise ValueError(f"No classes found in directory '{data_path}'.")
    for label in classes:
        class_dir = os.path.join(data_path, label)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Can't read image {img_path}, skipping.")
                continue
            img = cv2.resize(img, image_size)
            X.append(img)
            y.append(label)
    X = np.array(X).astype(np.float32) / 255.0
    # Normalisation centrée (mean=0, std=1) par pixel
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    y = np.array(y)
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    return X.reshape(len(X), -1), y_encoded, lb, classes


class MLPClassifier:
    def __init__(self, input_dim, hidden1, hidden2, output_dim, lr=0.001, reg_lambda=0.001, batch_size=64):
        # He initialization pour ReLU
        self.params = {
            'W1': np.random.randn(input_dim, hidden1) * np.sqrt(2. / input_dim),
            'b1': np.zeros((1, hidden1)),
            'W2': np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1),
            'b2': np.zeros((1, hidden2)),
            'W3': np.random.randn(hidden2, output_dim) * np.sqrt(2. / hidden2),
            'b3': np.zeros((1, output_dim)),
        }
        self.lr = lr
        self.initial_lr = lr
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.adam_params = {}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        for key in self.params:
            self.adam_params['m_' + key] = np.zeros_like(self.params[key])
            self.adam_params['v_' + key] = np.zeros_like(self.params[key])

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        W1, b1, W2, b2, W3, b3 = self.params.values()
        Z1 = X @ W1 + b1
        A1 = self.relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = self.relu(Z2)
        Z3 = A2 @ W3 + b3
        A3 = self.softmax(Z3)
        cache = (X, Z1, A1, Z2, A2, Z3, A3)
        return A3, cache

    def compute_loss(self, Y_hat, Y):
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        reg_loss = self.reg_lambda * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)) / m
        return loss + reg_loss

    def adam_update(self, grads):
        self.t += 1
        for key in self.params:
            m = self.adam_params['m_' + key]
            v = self.adam_params['v_' + key]
            g = grads[key]
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def backward(self, cache, Y):
        X, Z1, A1, Z2, A2, Z3, A3 = cache
        m = X.shape[0]
        grads = {}

        dZ3 = (A3 - Y) / m
        grads['W3'] = A2.T @ dZ3 + self.reg_lambda * self.params['W3'] / m
        grads['b3'] = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ self.params['W3'].T
        dZ2 = dA2 * self.relu_derivative(Z2)
        grads['W2'] = A1.T @ dZ2 + self.reg_lambda * self.params['W2'] / m
        grads['b2'] = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.params['W2'].T
        dZ1 = dA1 * self.relu_derivative(Z1)
        grads['W1'] = X.T @ dZ1 + self.reg_lambda * self.params['W1'] / m
        grads['b1'] = np.sum(dZ1, axis=0, keepdims=True)

        self.adam_update(grads)

    def predict(self, X):
        Y_hat, _ = self.forward(X)
        return np.argmax(Y_hat, axis=1)

    def plot_weight_histograms(self, epoch):
        plt.figure(figsize=(15, 4))
        for i, key in enumerate(['W1', 'W2', 'W3']):
            plt.subplot(1, 3, i + 1)
            plt.hist(self.params[key].flatten(), bins=50, color='c')
            plt.title(f'Histogram of {key} weights - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'weights_hist_epoch_{epoch}.png')
        plt.close()

    def visualize_predictions(self, X_batch, Y_batch, lb, epoch, n_samples=10):
        Y_pred_probs, _ = self.forward(X_batch)
        Y_pred = np.argmax(Y_pred_probs, axis=1)
        Y_true = np.argmax(Y_batch, axis=1)

        fig, axes = plt.subplots(2, n_samples, figsize=(20, 5))
        for i in range(n_samples):
            img = X_batch[i].reshape(32, 32, 3)
            # dé-normaliser pour afficher correctement (approx)
            img = (img * np.std(X_batch, axis=0).reshape(32, 32, 3)) + np.mean(X_batch, axis=0).reshape(32, 32, 3)
            img = np.clip(img, 0, 1)
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            axes[0, i].set_title(f"True: {lb.classes_[Y_true[i]]}")
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Pred: {lb.classes_[Y_pred[i]]}")
        plt.suptitle(f"Predictions at Epoch {epoch}")
        plt.savefig(f'predictions_epoch_{epoch}.png')
        plt.close()

    def fit(self, X, Y, X_val=None, Y_val=None, epochs=100):
        n_samples = X.shape[0]
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        print("=== Test forward + backward sur un batch ===")
        X_batch = X[:self.batch_size]
        Y_batch = Y[:self.batch_size]
        Y_hat, cache = self.forward(X_batch)
        loss = self.compute_loss(Y_hat, Y_batch)
        print(f"Loss batch: {loss:.4f}")
        self.backward(cache, Y_batch)
        print("Backward pass réussie sur batch")

        for epoch in range(epochs):
            # Decay learning rate every 20 epochs
            if epoch > 0 and epoch % 20 == 0:
                self.lr *= 0.7
                print(f"Learning rate decayed to {self.lr:.6f}")

            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]

            batch_losses = []

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                Y_batch = Y_shuffled[i:i+self.batch_size]
                Y_hat, cache = self.forward(X_batch)
                batch_loss = self.compute_loss(Y_hat, Y_batch)
                batch_losses.append(batch_loss)
                self.backward(cache, Y_batch)

            Y_hat_full, _ = self.forward(X)
            loss = self.compute_loss(Y_hat_full, Y)
            acc = np.mean(np.argmax(Y_hat_full, axis=1) == np.argmax(Y, axis=1))

            train_losses.append(loss)
            train_accuracies.append(acc)

            if X_val is not None and Y_val is not None:
                Y_val_hat, _ = self.forward(X_val)
                val_loss = self.compute_loss(Y_val_hat, Y_val)
                val_acc = np.mean(np.argmax(Y_val_hat, axis=1) == np.argmax(Y_val, axis=1))
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

            if (epoch + 1) % 10 == 0:
                avg_batch_loss = np.mean(batch_losses)
                print(f"Epoch {epoch+1}: Avg Batch Loss={avg_batch_loss:.4f}, Train Loss={loss:.4f}, Train Acc={acc*100:.2f}%")
                if X_val is not None and Y_val is not None:
                    print(f"Validation Loss={val_loss:.4f}, Validation Acc={val_acc*100:.2f}%")

                # Visualisation histogramme poids
                self.plot_weight_histograms(epoch + 1)

                # Visualisation prédictions sur un batch (dernier batch du shuffle)
                self.visualize_predictions(X_batch, Y_batch, lb, epoch + 1, n_samples=10)

        # Courbes globales loss / accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Loss curve")

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        if val_accuracies:
            plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title("Accuracy curve")

        plt.savefig('training_curves.png')
        print("Training curves saved to 'training_curves.png'")


if __name__ == "__main__":
    X, Y, lb, classes = load_data('data/AMHCD_64')

    print("Loaded data:", X.shape, Y.shape)
    print(f"Classes: {classes}")
    print(f"Exemple X[0] min={X[0].min()}, max={X[0].max()}, shape={X[0].shape}")
    print(f"Label Y[0] one-hot: {Y[0]}")
    print(f"Classe Y[0] inverse transform: {lb.inverse_transform(Y[0].reshape(1, -1))}")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    input_dim = X.shape[1]
    model = MLPClassifier(input_dim=input_dim, hidden1=64, hidden2=32, output_dim=len(classes), lr=0.001, batch_size=64)
    model.fit(X_train, Y_train, X_val, Y_val, epochs=100)

    joblib.dump(model, 'mlp_tifinagh_rgb_adam_debug.pkl')
    joblib.dump(lb, 'label_binarizer.pkl')

    # === Confusion Matrix ===
    Y_val_pred = model.predict(X_val)
    Y_val_true = np.argmax(Y_val, axis=1)

    cm = confusion_matrix(Y_val_true, Y_val_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(xticks_rotation=90, cmap='Blues', ax=ax)
    plt.title("Confusion Matrix - Validation Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved to 'confusion_matrix.png'")