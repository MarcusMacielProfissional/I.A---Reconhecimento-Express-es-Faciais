import cv2
import numpy as np
import tensorflow as tf
import time
import psutil
import multiprocessing
import tkinter as tk
from tkinter import Label
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
from threading import Thread

# 📌 Inicializar NVML para monitoramento da GPU
try:
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)
    gpu_available = True
except:
    gpu_available = False

# 📌 Função para monitoramento em uma Thread (evita problemas com Multiprocessing no Windows)
def monitoramento(fps):
    def atualizar_monitor():
        while True:
            uso_cpu_programa = psutil.Process().cpu_percent()
            label_cpu_programa.config(text=f"CPU (Programa): {uso_cpu_programa:.1f}%")

            uso_cpu_total = psutil.cpu_percent()
            label_cpu_total.config(text=f"CPU (Total): {uso_cpu_total:.1f}%")

            if gpu_available:
                uso_gpu = nvmlDeviceGetUtilizationRates(gpu_handle)
                label_gpu_programa.config(text=f"GPU (Programa): {uso_gpu.gpu:.1f}%")
                label_gpu_total.config(text=f"GPU (Total): {uso_gpu.gpu:.1f}%")
            else:
                label_gpu_programa.config(text="GPU (Programa): N/A")
                label_gpu_total.config(text="GPU (Total): N/A")

            label_fps.config(text=f"FPS: {int(fps.value)}")
            monitor.update_idletasks()
            time.sleep(1)

    # Criar janela Tkinter
    monitor = tk.Tk()
    monitor.title("Monitoramento de Desempenho")
    monitor.geometry("300x200")

    # Criar Labels
    label_cpu_programa = Label(monitor, text="CPU (Programa): --%", font=("Arial", 12))
    label_cpu_programa.pack()

    label_cpu_total = Label(monitor, text="CPU (Total): --%", font=("Arial", 12))
    label_cpu_total.pack()

    label_gpu_programa = Label(monitor, text="GPU (Programa): --%", font=("Arial", 12))
    label_gpu_programa.pack()

    label_gpu_total = Label(monitor, text="GPU (Total): --%", font=("Arial", 12))
    label_gpu_total.pack()

    label_fps = Label(monitor, text="FPS: --", font=("Arial", 12))
    label_fps.pack()

    # Rodar a atualização em uma Thread separada
    thread_atualizacao = Thread(target=atualizar_monitor, daemon=True)
    thread_atualizacao.start()

    monitor.mainloop()

# 📌 Função para iniciar a webcam
def iniciar_webcam(usar_gpu, fps):
    if usar_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("🚀 Usando GPU!")
        else:
            print("⚠️ Nenhuma GPU encontrada! Rodando na CPU...")
    else:
        tf.config.set_visible_devices([], 'GPU')
        print("💻 Usando apenas CPU.")

    modelo = tf.keras.models.load_model("modelo_emocoes.h5")
    emocoes = ["Raiva", "Nojo", "Medo", "Feliz", "Neutro", "Triste", "Surpreso"]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    ultimo_tempo = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (frame.shape[1] // 2, frame.shape[0] // 2))
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            x, y, w, h = x * 2, y * 2, w * 2, h * 2

            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
            face_roi = np.expand_dims(face_roi, axis=0).reshape(-1, 48, 48, 1)

            previsao = modelo.predict(face_roi, verbose=0)
            emocao_detectada = emocoes[np.argmax(previsao)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emocao_detectada, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        tempo_atual = time.time()
        fps.value = 1 / (tempo_atual - ultimo_tempo)
        ultimo_tempo = tempo_atual

        cv2.imshow("Reconhecimento de Emoções", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# 📌 Função para escolher o modo (CPU/GPU)
def escolher_modo():
    root = tk.Tk()
    root.title("Escolha o Modo de Execução")
    root.geometry("300x150")

    def definir_modo(modo):
        global usar_gpu
        usar_gpu = (modo == "gpu")
        root.destroy()  # Fecha a janela de escolha

    label = tk.Label(root, text="Escolha o modo de processamento:", font=("Arial", 12))
    label.pack(pady=10)

    btn_cpu = tk.Button(root, text="CPU Mode", font=("Arial", 10), command=lambda: definir_modo("cpu"), width=15)
    btn_cpu.pack(pady=5)

    btn_gpu = tk.Button(root, text="GPU Mode", font=("Arial", 10), command=lambda: definir_modo("gpu"), width=15)
    btn_gpu.pack(pady=5)

    root.mainloop()

# 📌 Função para iniciar os processos
def iniciar_processos():
    with multiprocessing.Manager() as manager:
        fps = manager.Value("d", 0.0)

        processo_webcam = multiprocessing.Process(target=iniciar_webcam, args=(usar_gpu, fps))
        thread_monitoramento = Thread(target=monitoramento, args=(fps,), daemon=True)

        processo_webcam.start()
        thread_monitoramento.start()

        processo_webcam.join()

# 📌 Executar apenas no processo principal
if __name__ == "__main__":
    escolher_modo()  # Pergunta ao usuário se quer CPU ou GPU
    iniciar_processos()  # Inicia os processos depois da escolha
