import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Configuración de semilla para reproducibilidad
np.random.seed(42)

# Datos de entrada y salida XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Hiperparámetros
learning_rate = 0.1
tasa_aprendizaje = 0.1
epocas = 10000

# Inicialización de pesos y sesgos
W1 = np.random.randn(2, 2) * 0.1
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1) * 0.1
b2 = np.zeros((1, 1))

# Entrenamiento de la red
for epoch in range(epocas):
    # Paso hacia adelante (Forward pass)
    entrada_oculta = np.dot(X, W1) + b1
    salida_oculta = np.tanh(entrada_oculta)
    capa_salida = np.dot(salida_oculta, W2) + b2
    salida = sigmoid(capa_salida)
    
    # Cálculo de pérdida
    perdida = np.mean(0.5 * (y - salida) ** 2)
    
    # Retropropagación (Backpropagation)
    delta3 = (salida - y) * salida * (1 - salida)
    dW2 = np.dot(salida_oculta.T, delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    
    delta2 = np.dot(delta3, W2.T) * (1 - salida_oculta ** 2)
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0, keepdims=True)
    
    # Actualización de parámetros
    W1 -= tasa_aprendizaje * dW1
    b1 -= tasa_aprendizaje * db1
    W2 -= tasa_aprendizaje * dW2
    b2 -= tasa_aprendizaje * db2
    
    # Mostrar pérdida cada 1000 épocas
    if epoch % 1000 == 0:
        print(f'Época {epoch}, Pérdida: {perdida:.4f}')

# Predicción después del entrenamiento
salida_oculta = np.tanh(np.dot(X, W1) + b1)
salida_final = sigmoid(np.dot(salida_oculta, W2) + b2)
print("\nSalida Predicha:")
print(salida_final)
