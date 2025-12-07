# Plano de IA

Esse é o plano de estudos completo para minhas férias de verão de 2026, meu foco é restruturar completamente os meus estudos de IA que ficaram desorganizados pelos ultimos meses.

Objetivo: Construir intuição matemática formal implementando algoritmos do zero (NumPy) e ganhar proficiência em ferramentas de mercado (PyTorch/Scikit-Learn).

---

## 📂 Estrutura do Repositório

*OBS: Organize seu código assim desde o Dia 1. Não use notebooks para código final.*


```
ai-studies/
├── README.md               # Seu diário de bordo.
├── requirements.txt        # Dependências (numpy, pandas, torch, black, pytest).
├── data/                   # (.gitignore) Dados brutos e processados.
├── notebooks/              # Apenas para exploração e gráficos (Rascunho).
├── src/                    # Código de Produção (Classes, Funções).
│   ├── models/             # Suas implementações (knn.py, linear_reg.py, mlp.py).
│   ├── metrics/            # Suas métricas (accuracy.py, f1.py).
│   ├── preprocessing/      # Normalizadores e DataLoaders.
│   └── visualization/      # Plots de curvas de aprendizado.
└── tests/                  # Testes unitários (test_linear.py, test_mlp.py).
```

---

##  Módulo 1: Matemática & Algoritmos Clássicos (NumPy Puro)

**Restrições:** Proibido usar Scikit-Learn para modelos. Apenas NumPy, Pandas e Matplotlib.

### Submódulo 1.1: O Início (KNN & Métricas)

*Foco: Entender distâncias vetoriais e como medir sucesso.*

- [ ] **Engenharia de Software**
  
  - [ ] Configurar ambiente (venv), Git e instalar `black` (linter) e `pytest`.
    
  - [X] Criar estrutura de pastas.
    
- [X] **Modelo: K-Nearest Neighbors (KNN)**
  
  - [X] Criar classe `KNNClassifier` em `src/models/knn.py`.
    
  - [ X Implementar `.fit(X, y)` (Apenas armazena dados).
    
  - [X] Implementar `.predict(X)` usando **Broadcasting do NumPy** para calcular Distância Euclidiana ($\sqrt{\sum(x-y)^2}$) sem loops `for`.
    
- [ ] **Métricas (Do Zero)**
  
  - [ ] Implementar `accuracy_score`.
    
  - [ ] Implementar `confusion_matrix` (Entender TP, TN, FP, FN).
    

### Submódulo 1.2: O Motor da Otimização (Regressão Linear)

*Foco: O algoritmo mais importante de todos (Gradiente Descendente).*

- [ ] **Pré-processamento**
  
  - [ ] Implementar **Normalização (MinMax)** e **Padronização (Z-Score)** em `src/preprocessing`.
    
  - [ ] *Conceito:* Testar treinar sem normalizar e ver o modelo falhar.
    
- [ ] **Modelo: Regressão Linear**
  
  - [ ] Criar classe `LinearRegression`.
    
  - [ ] Implementar o loop de **Gradiente Descendente** ($\theta = \theta - \alpha \cdot \nabla J$), usando Batches.
    
  - [ ] Criar teste unitário comparando seus pesos finais com `sklearn.linear_model.LinearRegression`.
    
- [ ] **Avaliação**
  
  - [ ] Implementar **MSE** (Mean Squared Error) e **R²**.
    
  - [ ] Plotar a curva de Loss x Épocas (para ver o aprendizado acontecer).
    

### Submódulo 1.3: Classificação Probabilística (Regressão Logística)

*Foco: Probabilidade e Fronteiras de Decisão.*

- [ ] **Modelo: Regressão Logística**
  
  - [ ] Implementar função **Sigmoid** ($1 / (1 + e^{-z})$).
    
  - [ ] Implementar **Log-Loss / Binary Cross-Entropy** (A função de custo para Sim/Não).
    
- [ ] **Métricas Avançadas**
  
  - [ ] Implementar **Precision**, **Recall** e **F1-Score** (Crucial para dados desbalanceados).
    
  - [ ] *Desafio:* Entender o Trade-off: aumentar o threshold aumenta Precision mas diminui Recall.
    

### Submódulo 1.4: Não Supervisionado (Clustering)

*Foco: Trabalhar sem labels.*

- [ ] **Modelo: K-Means**
  
  - [ ] Implementar inicialização de centróides.
    
  - [ ] Implementar loop: 1. Atribuir pontos ao centróide mais próximo; 2. Mover centróide para a média dos pontos.
    
  - [ ] Gerar um GIF ou plot mostrando os centróides "andando" até o centro dos clusters.
    

---

##  Módulo 2: Deep Learning & A Transição (NumPy -> PyTorch)

**Foco:** Sair da estatística linear e entrar na não-linearidade (Redes Neurais).

### Submódulo 2.1: O Chefão Final do NumPy (MLP)

*O desafio mais difícil do curso.*

- [ ] **Matemática "Hardcore"**
  
  - [ ] Entender a **Regra da Cadeia** (Chain Rule) para derivadas compostas.
- [ ] **Modelo: Multi-Layer Perceptron (Do Zero)**
  
  - [ ] Criar classe `NeuralNetwork`.
    
  - [ ] Implementar **Forward Pass** (Multiplicação de Matrizes + Ativação ReLU).
    
  - [ ] Implementar **Backpropagation** (Calcular gradientes camada por camada voltando do fim para o começo).
    
  - [ ] Resolver o problema do **XOR** (Dataset não linear).
    
- [ ] **Conceitos de Treino**
  
  - [ ] Implementar suporte a **Mini-Batches** (Não passar o dataset inteiro de uma vez).

### Submódulo 2.2: Bem-vindo ao PyTorch

*Foco: Traduzir o que você fez acima para a ferramenta profissional.*

- [ ] **Ferramental**
  
  - [ ] Entender **Tensores** (GPU) e **Autograd** (Derivada Automática - adeus regra da cadeia manual).
- [ ] **Refatoração**
  
  - [ ] Recriar a MLP acima usando `torch.nn.Linear` e `torch.optim.Adam`.
    
  - [ ] Comparar a velocidade de convergência do seu otimizador manual vs Adam.
    

### Submódulo 2.3: Visão Computacional (CNNs)

*Foco: Processamento de Imagens e Dados Espaciais.*

- [ ] **Teoria & Prática**
  
  - [ ] Entender **Convolução** (Filtros) e **Pooling**.
    
  - [ ] Treinar uma CNN simples no dataset MNIST (Dígitos).
    
- [ ] **Engenharia de Dados (Crítico)**
  
  - [ ] Criar classes `Dataset` e `DataLoader` customizadas no PyTorch.
    
  - [ ] Implementar **Data Augmentation** (Rotacionar/Inverter imagens para evitar overfitting).
    
  - [ ] Salvar o melhor modelo durante o treino (`Model Checkpointing`).
    

---

##  Módulo 3: O Mundo Real (Finanças, NLP & SOTA)

**Foco:** Ferramentas de produção e arquiteturas modernas.

### Submódulo 3.1: Dados Tabulares & Finanças (Ensembles)

*Foco: O que ganha competições no Kaggle e domina o mercado financeiro.*

- [ ] **Teoria (Sem implementar do zero)**
  
  - [ ] Entender Árvores de Decisão (Entropia).
    
  - [ ] Entender Bagging (Random Forest) vs Boosting (XGBoost/LightGBM).
    
- [ ] **Prática Financeira**
  
  - [ ] Baixar dados de ações (Yahoo Finance).
    
  - [ ] Treinar um **XGBoost** para prever direção do mercado.
    
  - [ ] **Prevenção de Erros:** Garantir validação temporal (Time Series Split) e não aleatória.
    
  - [ ] **Explicabilidade:** Usar **SHAP Values** para entender quais features o modelo usou.
    

### Submódulo 3.2: Séries Temporais (RNNs/LSTMs)

*Foco: Memória e Sequências.*

- [ ] **Engenharia de Features**
  
  - [ ] Implementar **Janelamento (Sliding Window)** para transformar série temporal em aprendizado supervisionado.

- [ ] **Modelo: LSTM/GRU (PyTorch)**
  
  - [ ] Implementar uma rede recorrente para prever o próximo valor da série.
    
  - [ ] Comparar performance: LSTM (Deep Learning) vs XGBoost (Módulo 3.1).
    

### Submódulo 3.3: NLP Moderno & Transformers

*Foco: A arquitetura do ChatGPT.*

- [ ] **Matemática da Atenção**
  
  - [ ] Implementar a fórmula de **Self-Attention** ($Attention(Q, K, V)$) com NumPy/PyTorch (apenas a função, não a rede toda).
- [ ] **Ecossistema Hugging Face**
  
  - [ ] Usar a biblioteca `transformers`.
    
  - [ ] Carregar um modelo pré-treinado (BERTimbau - BERT em PT).
    
  - [ ] Fazer **Fine-Tuning** para classificação de sentimentos em manchetes financeiras.
    

### Submódulo 3.4: Finalização & Deploy Básico

- [ ] **Ciclo de Vida**
  
  - [ ] Criar script `inference.py` que carrega o modelo salvo (`.pt` ou `.pkl`) e recebe novos dados.

  - [ ] (Opcional) Containerizar com **Docker** simples.
    

---

###  Definição de Concluído (DoD)

Ao final de cada submódulo, você deve ter:

1. O arquivo `.py` do modelo na pasta `src/models`.  
2. Um script de teste em `tests/` que passa ("Green").
