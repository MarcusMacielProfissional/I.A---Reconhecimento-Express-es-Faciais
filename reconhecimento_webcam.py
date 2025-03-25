import cv2
import numpy as np
import tensorflow as tf

# Carregar o modelo salvo
modelo = tf.keras.models.load_model("modelo_emocoes.h5")

# Lista de emoções
emocoes = ["Raiva", "Nojo", "Medo", "Feliz", "Neutro", "Triste", "Surpreso"]

# Iniciar a webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
        face_roi = np.expand_dims(face_roi, axis=0).reshape(-1, 48, 48, 1)

        previsao = modelo.predict(face_roi)
        emocao_detectada = emocoes[np.argmax(previsao)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emocao_detectada, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento de Emoções", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
