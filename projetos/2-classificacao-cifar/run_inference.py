import os
import sys
import io
import pickle

# Configurar stdout e stderr para usar UTF-8 e evitar erros de codificação no Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Projeto 2 — Inferência com o Modelo Otimizado (model.tflite)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar especificamente o "model.tflite" (o artefato de edge, não o
#      model.h5) usando tf.lite.Interpreter
#   2. Rodar inferência em pelo menos 5 amostras do conjunto de teste do CIFAR-10
#   3. Imprimir no terminal, para cada amostra: classe predita vs. classe real
# ---------------------------------------------------------------------------

N_SAMPLES = 5

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dataset_path = r"C:\Users\Wanderson\.keras\datasets\cifar-10-batches-py"

    with open(os.path.join(dataset_path, "test_batch"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    x_test = batch[b"data"]
    y_test = np.array(batch[b"labels"])

    x_test = x_test.reshape(-1, 3, 32, 32)
    x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype("float32") / 255.0
    
    N_SAMPLES = min(5, len(x_test))

    print(f"Rodando inferência em {N_SAMPLES} amostras usando model.tflite:\n")
    for i in range(N_SAMPLES):
        sample = np.expand_dims(x_test[i], axis=0).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0]
        predicted_class = int(np.argmax(pred))
        
        status = "✓" if predicted_class == y_test[i] else "✗"
        print(
            f"Amostra {i+1}: "
            f"Predito = {CLASS_NAMES[predicted_class]} | "
            f"Real = {CLASS_NAMES[int(y_test[i])]} "
            f"{status}"
        )


if __name__ == "__main__":
    main()
