# tests/test_knn.py
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from src.models.knn import KNNClassifier  

def test_knn_match_sklearn():
    """
    Testando a capacidade do meu modelo proprio de kNN contra o modelo completo do Scikit-learn 
    """
    # 1. Gerar dados
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    # 2. Modelo do Scikit-Learn
    sk_model = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
    sk_model.fit(X, y)
    expected_preds = sk_model.predict(X)
    
    # 3. Meu modelo
    my_model = KNNClassifier(k=3)
    my_model.fit(X, y)
    actual_preds = my_model.predict(X)
    
    # 4. A Prova Real
    assert np.array_equal(actual_preds, expected_preds)
    
    #5. Testar se a acurácia 
    assert np.mean(actual_preds == y) > 0.9