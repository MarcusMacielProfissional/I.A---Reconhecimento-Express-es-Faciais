import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Caminhos para os dados
train_dir = "C:/Users/Admin/Documents/Vini/datasets/train"
test_dir = "C:/Users/Admin/Documents/Vini/datasets/test"

# Mapeamento das emoções
emo_map = {
    "angry": 0, "disgust": 1, "fear": 2,
    "happy": 3, "neutral": 4, "sad": 5, "surprise": 6
}

# Função para carregar os dados
def carregar_dados(diretorio):
    imagens, labels = [], []
    
    for pasta in os.listdir(diretorio):
        pasta_path = os.path.join(diretorio, pasta)
        
        if pasta in emo_map:
            for imagem_nome in os.listdir(pasta_path):
                imagem_path = os.path.join(pasta_path, imagem_nome)
                if imagem_nome.endswith('.jpg'):
                    imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
                    imagem = cv2.resize(imagem, (48, 48)) / 255.0  
                    imagens.append(imagem)
                    labels.append(emo_map[pasta])
    
    imagens = np.array(imagens).reshape(-1, 48, 48, 1)
    labels = to_categorical(labels, num_classes=7)
    return imagens, labels

# Carregar os dados
X_train, y_train = carregar_dados(train_dir)
X_test, y_test = carregar_dados(test_dir)

# Criar o modelo
modelo = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")
])

modelo.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Treinar o modelo
print("Treinando o modelo...")
modelo.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Salvar o modelo treinado
modelo.save("modelo_emocoes.h5")
print("✅ Modelo treinado e salvo como 'modelo_emocoes.h5'!")
