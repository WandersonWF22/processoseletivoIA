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

# ---------------------------------------------------------------------------
# Projeto 2 — Otimização do Modelo (CIFAR-10)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui
