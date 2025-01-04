import os
import librosa
import numpy as np

# Diccionario que mapea cada genero a un indice numerico
genero_a_indice = {
    'rock': 0,
    'blues': 1,
    'classical': 2,
    'country': 3,
    'disco': 4,
    'hiphop': 5,
    'jazz': 6,
    'metal': 7,
    'pop': 8,
    'reggae': 9
}

def procesar_audio_y_generar_dataset(carpeta_audio, genero_a_indice, tamano_maximo=128, n_mels=128):
    dataset = []
    etiquetas = []

    for archivo in os.listdir(carpeta_audio):
        if archivo.endswith('.mp3') or archivo.endswith('.wav'):
            audio_path = os.path.join(carpeta_audio, archivo)
            y, sr = librosa.load(audio_path)
            
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            if S_dB.shape[1] > tamano_maximo:
                S_dB = S_dB[:, :tamano_maximo]  
            elif S_dB.shape[1] < tamano_maximo:
                pad_width = tamano_maximo - S_dB.shape[1]
                S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant')

            S_dB = np.expand_dims(S_dB, axis=-1)
            dataset.append(S_dB)
            
            genero = archivo.split('.')[0].lower()
            indice = genero_a_indice.get(genero, -1)
            if indice != -1:
                etiquetas.append(indice)

    # Convertir las listas a arreglos numpy
    dataset = np.array(dataset)
    etiquetas = np.array(etiquetas)

    return dataset, etiquetas

# Buscar carpetas con url relativas
directorio_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_entrenamiento = os.path.join(directorio_actual, 'train')
carpeta_prueba = os.path.join(directorio_actual, 'test')

# Buscar y crear la carpeta Dataset
carpeta_salida = os.path.join(directorio_actual, 'Dataset')
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)


try:
    # Procesar audios y generar datasets 
    X_train, Y_train = procesar_audio_y_generar_dataset(carpeta_entrenamiento, genero_a_indice)
    X_test, Y_test = procesar_audio_y_generar_dataset(carpeta_prueba, genero_a_indice)

    # Guardar los datasets y etiquetas
    np.save(os.path.join(carpeta_salida, 'X_train.npy'), X_train)
    np.save(os.path.join(carpeta_salida, 'Y_train.npy'), Y_train)
    np.save(os.path.join(carpeta_salida, 'X_test.npy'), X_test)
    np.save(os.path.join(carpeta_salida, 'Y_test.npy'), Y_test)

    print("El procesamiento de datos ha finalizado correctamente.")
    
except Exception as e:
    print(f"Error en el procesamiento de datos: {e}")

