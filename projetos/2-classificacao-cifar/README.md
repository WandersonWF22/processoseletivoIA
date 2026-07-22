## 📝 Relatório do Candidato

👤 **Nome Completo:** Wanderson Wilkerson Felix da Silva

### 1️⃣ Resumo da Arquitetura do Modelo

Foi desenvolvida uma Rede Neural Convolucional (CNN) para classificação de imagens do conjunto CIFAR-10. O modelo recebe imagens RGB de dimensão 32×32 pixels normalizadas no intervalo [0,1]. A arquitetura é composta por uma camada de entrada seguida por um bloco de Data Augmentation, responsável por gerar variações das imagens durante o treinamento. Em seguida, o modelo possui três blocos convolucionais formados por camadas `Conv2D`, `BatchNormalization` e `MaxPooling2D`, responsáveis pela extração progressiva das características das imagens.

Após a etapa convolucional, as características são convertidas para um vetor utilizando a camada `Flatten`, seguido por uma camada densa com 128 neurônios e função de ativação ReLU. Antes da camada de saída foi utilizada uma camada `Dropout` com taxa de 0,5 para reduzir o risco de overfitting. A camada final possui 10 neurônios com ativação Softmax, correspondendo às dez classes do dataset CIFAR-10.

Como estratégia de Data Augmentation foram utilizadas as camadas `RandomFlip("horizontal")`, `RandomRotation(0.1)` e `RandomZoom(0.1)`, permitindo que pequenas variações fossem aplicadas automaticamente às imagens durante o treinamento. Essa técnica contribui para melhorar a capacidade de generalização do modelo e reduzir o sobreajuste.

### 2️⃣ Bibliotecas Utilizadas

As principais bibliotecas utilizadas foram:

* TensorFlow 2.21.0
* Keras 3.15.0
* NumPy 2.4.6
* Biblioteca padrão do Python (`os`, `io` e `sys`)

Essas bibliotecas foram utilizadas para construção da CNN, treinamento, otimização do modelo, manipulação dos dados e execução da inferência.

### 3️⃣ Técnica de Otimização do Modelo

Após o treinamento, o modelo salvo em formato Keras (`model.h5`) foi convertido para TensorFlow Lite utilizando a classe `tf.lite.TFLiteConverter`. Durante a conversão foi aplicada a técnica de **Dynamic Range Quantization**, através da configuração:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

Essa técnica reduz o tamanho do modelo e melhora sua eficiência para execução em dispositivos de borda, mantendo um bom nível de precisão sem necessidade de retreinamento.

###

### 4️⃣ Resultados Obtidos

O treinamento foi realizado utilizando Early Stopping baseado na perda do conjunto de validação, permitindo interromper automaticamente o treinamento quando não havia melhora significativa.

Durante os testes locais realizados no ambiente de desenvolvimento, o modelo apresentou uma acurácia de validação aproximada de **68,15%**.

Após a execução no ambiente de integração contínua do GitHub Actions, uma nova execução de treinamento apresentou uma acurácia de validação de **75,56%**. 

Os arquivos gerados apresentaram os seguintes tamanhos:

* `model.h5`: **4,16 MB**
* `model.tflite`: **365 KB**

A redução de tamanho obtida após a conversão demonstra a eficiência da técnica de otimização aplicada, tornando o modelo adequado para execução em dispositivos com recursos limitados.

### 5️⃣ Comentários Adicionais

Durante o desenvolvimento eu encontrei uma dificuldade relacionada ao carregamento automático do dataset CIFAR-10. No ambiente local, a função `tf.keras.datasets.cifar10.load_data()` apresentou falha devido a um problema de certificado SSL no servidor de hospedagem do dataset, impedindo o download automático do arquivo durante a execução dessa etapa de forma automática.

Como solução temporária para dar continuidade ao desenvolvimento e aos testes, realizei o carregamento manual do dataset a partir dos arquivos disponibilizados pelo próprio CIFAR-10. Posteriormente, ao identificar que essa abordagem não era compatível com a validação automática do GitHub Actions, eu ajustei o código para retornar ao método recomendado no desafio, utilizando `tf.keras.datasets.cifar10.load_data()`.

Também foi necessário revisar o código para garantir compatibilidade entre o ambiente local e o ambiente de integração contínua do GitHub Actions, removendo dependências de caminhos absolutos do sistema operacional e assegurando que todo o pipeline pudesse ser executado automaticamente.

Após esses ajustes, todos os requisitos do desafio foram atendidos e a validação automática foi concluída com sucesso.

Essa experiência permitiu compreender melhor a importância da portabilidade do código, da compatibilidade entre diferentes ambientes de execução e da utilização de boas práticas durante o desenvolvimento de aplicações de Inteligência Artificial voltadas para dispositivos Edge.

### 6️⃣ Exemplo de Inferência

Saída obtida durante a execução de `run_inference.py`:

```text
Amostra 1: Predito = dog | Real = cat ✗
Amostra 2: Predito = ship | Real = ship ✓
Amostra 3: Predito = ship | Real = ship ✓
Amostra 4: Predito = airplane | Real = airplane ✓
Amostra 5: Predito = frog | Real = frog ✓
```

Nas cinco primeiras amostras, o modelo obteve quatro classificações corretas e uma incorreta. O único erro ocorreu entre as classes "cat" e "dog", categorias visualmente semelhantes e frequentemente confundidas devido à baixa resolução das imagens (32×32 pixels). Os demais exemplos foram classificados corretamente, demonstrando que o modelo otimizado em TensorFlow Lite manteve sua capacidade de inferência após a conversão.

Na versão final do projeto, decidi aumentar a quantidade de amostras utilizadas na inferência de 5 para 10. Essa alteração teve como objetivo verificar novamente o funcionamento do modelo durante a execução da validação automática no GitHub Actions.

Eu percebi que a taxa de acerto observada pode variar conforme a quantidade e as amostras utilizadas na inferência, sendo apenas uma demonstração prática do funcionamento do modelo, e não sua acurácia geral.
