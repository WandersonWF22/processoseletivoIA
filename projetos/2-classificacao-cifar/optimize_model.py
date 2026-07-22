import tensorflow as tf
import os

# Carrega o modelo treinado
model = tf.keras.models.load_model("model.h5")

# Conversor TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Dynamic Range Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Converte o modelo
tflite_model = converter.convert()

# Salva
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo convertido com sucesso!")

print(f"Tamanho model.h5: {os.path.getsize('model.h5')/1024/1024:.2f} MB")
print(f"Tamanho model.tflite: {os.path.getsize('model.tflite')/1024/1024:.2f} MB")
