import numpy as np
import pandas as pd

# 1. Función para recolectar datos (simulación)
def recolectar_datos():
    # Datos de ejemplo (deberían ser reemplazados con datos reales)
    reportes = {'Vía A': 2, 'Vía B': 0}  # Reportes manuales de esta semana
    clima = {'precipitacion': 45, 'prob_lluvia': 70}  # Datos climáticos simulados
    mantenimiento = {'Vía A': 150, 'Vía B': 30}  # Días sin mantenimiento
    
    # Crear array de características para cada vía
    datos = []
    for via in ['Vía A', 'Vía B']:
        datos.append([
            clima['precipitacion'],
            mantenimiento[via],
            reportes.get(via, 0),
            1 if via == 'Vía A' else 0  # Tipo de terreno (ejemplo)
        ])
    
    return np.array(datos), ['Vía A', 'Vía B']

# 2. Clase MLP
class MLP:
    def __init__(self, input_size, hidden_size):
        # Inicialización de parámetros
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros(1)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, X):
        # Propagación hacia adelante
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output, lr=0.01):
        # Retropropagación
        m = X.shape[0]
        
        # Cálculo de gradientes
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0)
        
        dZ1 = np.dot(dZ2, self.W2.T) * (self.z1 > 0)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0)
        
        # Actualización de parámetros
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def train(self, X, y, epochs=1000, lr=0.01):
        # Normalización de datos
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X_norm = (X - self.mean) / self.std
        
        # Entrenamiento
        for epoch in range(epochs):
            output = self.forward(X_norm)
            loss = self.binary_crossentropy(y, output)
            self.backward(X_norm, y, output, lr)
            
            if epoch % 100 == 0:
                print(f'Época {epoch}, Pérdida: {loss:.4f}')
    
    def binary_crossentropy(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def predict(self, X):
        # Normalización usando parámetros de entrenamiento
        X_norm = (X - self.mean) / self.std
        return self.forward(X_norm)

# 3. Cargar datos históricos
# Estructura del CSV: precipitacion,dias_mantenimiento,reportes,terreno,averiada
datos_historicos = pd.DataFrame({
    'precipitacion': [80, 30, 45, 90, 20],
    'dias_mantenimiento': [120, 30, 90, 150, 15],
    'reportes': [3, 0, 2, 5, 0],
    'terreno': [1, 0, 1, 1, 0],
    'averiada': [1, 0, 1, 1, 0]
})

# 4. Entrenar el modelo
X_train = datos_historicos[['precipitacion', 'dias_mantenimiento', 'reportes', 'terreno']].values
y_train = datos_historicos['averiada'].values.reshape(-1, 1)

mlp = MLP(input_size=4, hidden_size=8)
mlp.train(X_train, y_train, epochs=2000, lr=0.01)

# 5. Sistema de predicción y rutas
rutas_alternas = {
    'Vía A': 'Ruta Alterna 1',
    'Vía B': 'Ruta Alterna 2'
}

def gestion_transporte():
    # Recolectar datos actuales
    datos_actuales, vias = recolectar_datos()
    
    # Predecir averías
    predicciones = mlp.predict(datos_actuales)
    
    # Actualizar rutas y notificar
    for via, prob in zip(vias, predicciones.flatten()):
        if prob > 0.7:  # Umbral de decisión
            print(f'[{via}] Probabilidad de avería: {prob*100:.1f}% - ACTIVAR {rutas_alternas[via]}')
            enviar_notificacion(via, rutas_alternas[via])
        else:
            print(f'[{via}] Probabilidad de avería: {prob*100:.1f}% - Ruta operativa')

def enviar_notificacion(via, ruta_alterna):
    # Simulación de envío de notificación
    print(f'Notificación: {via} en riesgo. Usar {ruta_alterna}\nEnviando SMS a conductores...\n')

# Ejecutar el sistema
if __name__ == '__main__':
    print("\n--- Sistema de Gestión de Transporte Poblado Boniato ---\n")
    gestion_transporte()
