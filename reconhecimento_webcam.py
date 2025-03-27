import cv2
import numpy as np
import tensorflow as tf
import psutil
import time
import threading
import tkinter as tk
import multiprocessing
import GPUtil

# Carregar o modelo salvo
modelo = tf.keras.models.load_model("modelo_emocoes.h5")

# Lista de emoções
emocoes = ["Raiva", "Nojo", "Medo", "Feliz", "Neutro", "Triste", "Surpreso"]

# Configuração global para modo CPU/GPU
modo_uso = multiprocessing.Value("i", 0)  # 0 = CPU, 1 = GPU

def escolher_modo():
    """Janela para selecionar CPU ou GPU antes de iniciar a câmera."""
    def selecionar_cpu():
        modo_uso.value = 0
        root.destroy()

    def selecionar_gpu():
        modo_uso.value = 1
        root.destroy()

    root = tk.Tk()
    root.title("Escolha o Modo de Execução")

    label = tk.Label(root, text="Escolha como deseja rodar o reconhecimento facial:")
    label.pack(pady=10)

    btn_cpu = tk.Button(root, text="Usar CPU", command=selecionar_cpu)
    btn_cpu.pack(pady=5)

    btn_gpu = tk.Button(root, text="Usar GPU", command=selecionar_gpu)
    btn_gpu.pack(pady=5)

    root.mainloop()

def obter_info_gpu():
    """Verifica se há GPUs disponíveis, incluindo integradas."""
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].name  # Retorna o nome da primeira GPU dedicada encontrada
    
    # Verifica via TensorFlow
    gpus_tf = tf.config.experimental.list_physical_devices('GPU')
    if gpus_tf:
        return gpus_tf[0].name
    
    # Caso não encontre GPUs, verifica gráficos integrados via psutil
    for proc in psutil.process_iter(['pid', 'name']):
        if 'intel' in proc.info['name'].lower() or 'amd' in proc.info['name'].lower():
            return proc.info['name']
    
    return None

def monitorar_desempenho():
    """Mostra o uso da CPU e GPU geral do sistema em uma janela separada."""
    root = tk.Tk()
    root.title("Desempenho do Programa")

    label_cpu_total = tk.Label(root, text="CPU Geral: 0%")
    label_cpu_total.pack(pady=5)

    label_gpu_total = tk.Label(root, text="GPU Geral: 0%")
    label_gpu_total.pack(pady=5)

    def atualizar():
        """Atualiza os valores de uso da CPU e GPU dinamicamente."""
        while True:
            cpu_total = psutil.cpu_percent(interval=1)  # Atualiza a cada 1s
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_total = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
            else:
                gpu_total = 0
            
            label_cpu_total.config(text=f"CPU Geral: {cpu_total:.1f}%")
            label_gpu_total.config(text=f"GPU Geral: {gpu_total:.1f}%")
            
            time.sleep(1)

    threading.Thread(target=atualizar, daemon=True).start()
    root.mainloop()

def iniciar_reconhecimento():
    """Inicia o reconhecimento facial com a webcam e faz previsões de emoções."""
    cap = cv2.VideoCapture(0)

    # Verifique se a webcam foi aberta corretamente
    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem.")
            break

        # Pre-processamento da imagem para o modelo
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)  # Convertendo a face para escala de cinza
            face_region = cv2.resize(face_region, (48, 48))  # Redimensiona para 48x48
            face_region = np.expand_dims(face_region, axis=-1)  # Adiciona o canal único (1)
            face_region = np.expand_dims(face_region, axis=0)  # Adiciona a dimensão do batch (1, 48, 48, 1)
            face_region = face_region / 255.0  # Normaliza a imagem

            # Fazer a previsão
            predicao = modelo.predict(face_region)
            emocao = emocoes[np.argmax(predicao)]

            # Desenhar a caixa ao redor do rosto e a emoção prevista
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emocao, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Reconhecimento Facial', frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Escolher CPU ou GPU antes de iniciar
    escolher_modo()

    # Verificar se há GPU disponível e configurar TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        gpu_nome = gpus[0].name
        print(f"GPU detectada: {gpu_nome}")
        if modo_uso.value == 1:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("Nenhuma GPU detectada. Usando CPU.")
    
    # Iniciar monitoramento de desempenho em um processo separado
    thread_desempenho = multiprocessing.Process(target=monitorar_desempenho)
    thread_desempenho.start()

    # Iniciar o reconhecimento facial em um processo separado
    thread_reconhecimento = multiprocessing.Process(target=iniciar_reconhecimento)
    thread_reconhecimento.start()

    # Esperar os processos terminarem
    thread_reconhecimento.join()
    thread_desempenho.terminate()
