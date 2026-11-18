
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


DATASET_DIR = "C:\\Users\\kanch\\OneDrive\\Desktop\\Project neural network"

TRAIN_IMAGES = os.path.join("C:\\Users\\kanch\\OneDrive\\Desktop\\Project neural network\\train-images.idx3-ubyte")
TRAIN_LABELS = os.path.join("C:\\Users\\kanch\\OneDrive\\Desktop\\Project neural network\\train-labels.idx1-ubyte")

TEST_IMAGES = os.path.join("C:\\Users\\kanch\\OneDrive\\Desktop\\Project neural network\\t10k-images(1).idx3-ubyte")
TEST_LABELS = os.path.join("C:\\Users\\kanch\\OneDrive\\Desktop\\Project neural network\\t10k-labels.idx1-ubyte")




def load_images(file_path):

    with open(file_path, "rb") as file:

        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))

        if magic != 2051:
            raise ValueError("Invalid MNIST image file: " + file_path)

        images = np.frombuffer(file.read(), dtype=np.uint8)

        images = images.reshape(count, rows * cols)

    return images



def load_labels(file_path):

    with open(file_path, "rb") as file:

        magic, count = struct.unpack(">II", file.read(8))

        if magic != 2049:
            raise ValueError("Invalid MNIST label file: " + file_path)

        labels = np.frombuffer(file.read(), dtype=np.uint8)

    if len(labels) != count:
        raise ValueError("Image label count does not match header")

    return labels

X_train = load_images(TRAIN_IMAGES)
y_train = load_labels(TRAIN_LABELS)

X_test = load_images(TEST_IMAGES)
y_test = load_labels(TEST_LABELS)

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)



X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0






plt.imshow(X_train[0].reshape(28, 28), cmap="gray")
plt.title("Actual digit: " + str(y_train[0]))
plt.axis("off")
plt.show()




np.random.seed(42)

input_neurons = 784
hidden_neurons = 128
output_neurons = 10


W1 = np.random.randn(input_neurons, hidden_neurons) * np.sqrt(2 / input_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons) * np.sqrt(2 / hidden_neurons)
b2 = np.zeros((1, output_neurons))




def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):

    Z = Z - np.max(Z, axis=1, keepdims=True)

    exp_values = np.exp(Z)

    probabilities = exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

    return probabilities




def forward_propagation(X):

    Z1 = np.dot(X, W1) + b1

    A1 = relu(Z1)


    Z2 = np.dot(A1, W2) + b2

    A2 = softmax(Z2)

    return Z1, A1, A2



def calculate_loss(predictions, labels):

    samples = len(labels)

    correct_probabilities = predictions[
        np.arange(samples), labels
    ]

    correct_probabilities = np.clip(
        correct_probabilities, 1e-12, 1.0
    )

    loss = -np.mean(np.log(correct_probabilities))

    return loss



def backward_propagation(X, labels, Z1, A1, A2):

    global W1, b1, W2, b2

    samples = X.shape[0]

    dZ2 = A2.copy()

    dZ2[np.arange(samples), labels] -= 1

    dZ2 = dZ2 / samples

    dW2 = np.dot(A1.T, dZ2)

    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)


    dZ1 = dA1 * (Z1 > 0)

    dW1 = np.dot(X.T, dZ1)

    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2




def predict(X):

    _, _, probabilities = forward_propagation(X)

    predictions = np.argmax(probabilities, axis=1)

    return predictions




def calculate_accuracy(X, labels):

    predictions = predict(X)

    correct = np.sum(predictions == labels)

    accuracy = correct / len(labels) * 100

    return accuracy


epochs = 10
batch_size = 128
learning_rate = 0.1

loss_history = []
accuracy_history = []

print("\nStarting neural network training...\n")

for epoch in range(epochs):


    indices = np.random.permutation(len(X_train))

    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    total_loss = 0


    for start in range(0, len(X_train), batch_size):

        end = start + batch_size

        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        Z1, A1, A2 = forward_propagation(X_batch)

        loss = calculate_loss(A2, y_batch)

        total_loss += loss * len(X_batch)

        dW1, db1, dW2, db2 = backward_propagation(
            X_batch, y_batch, Z1, A1, A2
        )

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    average_loss = total_loss / len(X_train)


    accuracy = calculate_accuracy(X_test, y_test)

    loss_history.append(average_loss)
    accuracy_history.append(accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {average_loss:.4f} | "
        f"Test Accuracy: {accuracy:.2f}%"
    )

3

final_accuracy = calculate_accuracy(X_test, y_test)

print("\nTraining completed!")
print("Final Test Accuracy:", round(final_accuracy, 2), "%")



np.savez(
    "trained_model.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)





plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(loss_history)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")


plt.subplot(1, 2, 2)

plt.plot(accuracy_history)

plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()
sample_predictions = predict(X_test[:10])

plt.figure(figsize=(12, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")

    plt.title(
        f"Actual: {y_test[i]}\nPredicted: {sample_predictions[i]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()