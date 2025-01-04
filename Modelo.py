import os
import numpy as np

np.random.seed(1)

def one_hot_encoding(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])  # Inicialización de He
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    Z = W.dot(A) + b    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    return Z

def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    
    return A

def softmax(Z):
    e = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    s = e / e.sum(axis=0, keepdims=True)
    return s

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "softmax":
        Z = linear_forward(A_prev, W, b)
        A = softmax(Z)

    elif activation == "relu":
        Z = linear_forward(A_prev, W, b)
        A = relu(Z)
    
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (Z, A_prev, W)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                

    for l in range(1, L):
        A_prev = A 
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)
    
    A_prev = A
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A_prev, W, b, "softmax")
    caches.append(cache)
    
    assert(AL.shape == (W.shape[0], X.shape[1])), f"Dimensiones no coinciden: AL {AL.shape}, se esperaba ({W.shape[0]}, {X.shape[1]})"
            
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]  # número de ejemplos
    AL = AL.reshape(Y.shape)  # Asegurarse de que AL tenga la misma forma que Y
    cost = -1/m * np.sum(Y * np.log(AL + 1e-8))
    cost = np.squeeze(cost)
    assert(AL.shape == Y.shape), f"Dimensiones no coinciden: AL {AL.shape}, Y {Y.shape}"
    return cost

def linear_backward(dZ, cache):
    Z, A_prev, W = cache 
    m = A_prev.shape[1]

    dW = 1/m * dZ.dot(A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = (W.T).dot(dZ)
    
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    Z, A_prev, W = cache
    
    if activation == "relu":
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        dA_prev, dW, db = linear_backward(dZ, cache)
    elif activation == "softmax":
        dZ = dA  # Usamos dA directamente como dZ = AL - Y
        dA_prev, dW, db = linear_backward(dZ, cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    dAL = AL - Y

    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
             linear_activation_backward(dAL, current_cache, "softmax")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
             linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate, lambd=0.0):
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * (grads["dW" + str(l+1)] + lambd * parameters["W" + str(l+1)])
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

    return parameters


# Cargar los datos
# Obtener el directorio actual del script
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta completa a la carpeta Dataset
carpeta_dataset = os.path.join(directorio_actual, 'Dataset')

# Cargar los archivos npy
X_train = np.load(os.path.join(carpeta_dataset, 'X_train.npy'))
Y_train = np.load(os.path.join(carpeta_dataset, 'Y_train.npy'))
X_test = np.load(os.path.join(carpeta_dataset, 'X_test.npy'))
Y_test = np.load(os.path.join(carpeta_dataset, 'Y_test.npy'))

# Verificar la forma de los datasets cargados
print("Forma de X_train:", X_train.shape)
print("Forma de Y_train:", Y_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de Y_test:", Y_test.shape)

# Aplanar los datos de entrada
train_x_flatten = X_train.reshape(X_train.shape[0], -1).T  # (n_instancias, dimensiones) -> (dimensiones, n_instancias)
test_x_flatten = X_test.reshape(X_test.shape[0], -1).T

# Normalizar los datos
X_train = train_x_flatten / 255.0
X_test = test_x_flatten / 255.0

print ("train_x's shape: " + str(X_train.shape))
print ("test_x's shape: " + str(X_test.shape))

# Convertir etiquetas a one-hot encoding
num_classes = 10  # Número de clases
Y_train = one_hot_encoding(Y_train, num_classes).T # Transponer para que cada columna represente una instancia
Y_test = one_hot_encoding(Y_test, num_classes).T

print("Primeras 10 filas de Y_train:")
print(Y_train)  # Muestra las primeras 10 filas de cada columna

print("\nPrimeras 10 filas de Y_test:")
print(Y_test) 

# Definir dimensiones de las capas
layer_dims = (X_train.shape[0], 64, 32, 20, 20, 10) 
print('layer_dims:', layer_dims)

# Entrenamiento del modelo
def train_model(X, Y, layer_dims, learning_rate=0.005, num_iterations=6000, print_cost=True):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layer_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate, lambd=0.01)
        if print_cost and i % 100 == 0:
            print(f"Costo después de la iteración {i}: {cost}")
            costs.append(cost)
    return parameters, costs

# Entrenar el modelo
parameters, costs = train_model(X_train, Y_train, layer_dims)



# Evaluar el modelo en el conjunto de prueba
predictions_train, _ = L_model_forward(X_train, parameters)
predictions_test, _ = L_model_forward(X_test, parameters)

# Evaluación y precisión en el conjunto de prueba
accuracy_train = np.mean(np.argmax(predictions_train, axis=0) == np.argmax(Y_train, axis=0))
accuracy_test = np.mean(np.argmax(predictions_test, axis=0) == np.argmax(Y_test, axis=0))
print(f"Precisión en entrenamiento: {accuracy_train * 100}%")
print(f"Precisión en prueba: {accuracy_test * 100}%")

# Función para predecir la clase de un audio dado su índice
def predecir_clase_audio(indice, X, parameters):
    X_audio = X[:, indice].reshape(-1, 1)
    prediccion, _ = L_model_forward(X_audio, parameters)
    clase_predicha = np.argmax(prediccion, axis=0)
    
    return clase_predicha[0]

# Ejemplo de uso: Predecir la clase de un audio del conjunto de prueba
indice_audio = 0  # Cambia este valor al índice que deseas predecir
clase_predicha = predecir_clase_audio(indice_audio, X_test, parameters)
print(f"El audio con índice {indice_audio} fue clasificado como clase {clase_predicha}")

