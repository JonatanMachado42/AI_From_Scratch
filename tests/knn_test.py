import numpy as np 
import sys
import os 
import matplotlib
matplotlib.use("TkAgg")  # ou Qt5Agg
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import sys
import os

# Pega o diretório atual onde o notebook está rodando
current_dir = os.getcwd()

# Sobe um nível (vai para a pasta pai, que é a raiz do projeto)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Adiciona ao path
sys.path.append(root_dir)

# Teste para ver se funcionou (deve imprimir o caminho da sua pasta ai-studies)
print(f"Raiz do projeto adicionada: {root_dir}")

# Agora tente importar
from src.models.knn import KNNClassifier




# Cria 300 pontos divididos em 4 grupos (clusters)
X_dataset, y_dataset = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
X_test, y_test = make_blobs(n_samples=60, centers=4, n_features=2, random_state=42)


knn = KNNClassifier(k=4)
knn.fit(X_dataset,y_dataset)
y_pred = knn.predict(X_test)

plt.figure(figsize=(6,5))

plt.scatter(X_dataset[:,0], X_dataset[:,1], c=y_dataset, cmap="viridis", s=40, label="Dataset")
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred,cmap="viridis",marker="^",s=70,edgecolors="black",label="Test")
plt.show()