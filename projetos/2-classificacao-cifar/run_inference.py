import os
import sys
import io

# Configurar stdout e stderr para usar UTF-8 e evitar erros de codificação no Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf

N_SAMPLES = 10  # Número de amostras para inferência

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

    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.reshape(-1)

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
