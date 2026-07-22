import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np

# Carregamento do dataset
import pickle

def load_cifar10(path):
    x_train = []
    y_train = []

    for i in range(1, 6):
        with open(os.path.join(path, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            x_train.append(batch[b"data"])
            y_train.extend(batch[b"labels"])

    x_train = np.concatenate(x_train)

    with open(os.path.join(path, "test_batch"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        x_test = batch[b"data"]
        y_test = batch[b"labels"]

    # CIFAR vem como (quantidade, 3072)
    # reorganizar para (quantidade, 32, 32, 3)
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return (x_train, np.array(y_train)), (x_test, np.array(y_test))


dataset_path = os.path.expanduser("~/.keras/datasets/cifar-10-batches-py")

if not os.path.exists(dataset_path):
    # Faz o download automático apenas se o dataset não existir
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
else:
    (x_train, y_train), (x_test, y_test) = load_cifar10(dataset_path)

# Normalização
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Separação explícita de treino e validação (10% para validação)
validation_size = int(len(x_train) * 0.1)

x_val = x_train[:validation_size]
y_val = y_train[:validation_size]

x_train = x_train[validation_size:]
y_train = y_train[validation_size:]

# Data Augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

# Construção do modelo
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),

    data_augmentation,

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(10, activation="softmax"),
])

# Compilação
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

# Treinamento
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=25,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1,
)

# Exibe acurácia final
final_accuracy = history.history["val_accuracy"][-1]
print(f"\nAcurácia final de validação: {final_accuracy:.4f}")

# Salva modelo
model.save("model.h5")

print("\nModelo salvo como model.h5")