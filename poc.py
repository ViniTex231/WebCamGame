import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Função para listar as câmeras disponíveis no PC
def listar_cameras():
    cameras = []
    for i in range(5):  # Tenta até 5 câmeras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
        cap.release()
    return cameras

# Função para carregar o modelo FER+ treinado (DCNN)
def carregar_modelo():
    modelo = tf.keras.models.load_model('modelo_ferplus.h5')  # Certifique-se de ter o modelo FER+ em .h5
    return modelo

# Função para processar a imagem e fazer a predição da expressão
def processar_expressao(frame, modelo, face_landmarks):
    bbox = face_landmarks['bbox']  # Pega a caixa delimitadora da face (bounding box)
    x, y, w, h = bbox
    face = frame[y:y+h, x:x+w]  # Extrair a região da face
    
    # Redimensionar a imagem para 48x48 (tamanho típico para redes de expressão facial)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza
    face = cv2.resize(face, (48, 48))  # Redimensionar para 48x48
    face = face / 255.0  # Normalizar
    face = np.reshape(face, (1, 48, 48, 1))  # Forma esperada para a entrada do modelo
    
    # Fazer a predição
    predicao = modelo.predict(face)
    expressao = np.argmax(predicao)  # Obter a expressão com a maior probabilidade
    
    # Mapear a expressão para o nome
    expressoes = ['Raiva', 'Desprezo', 'Medo', 'Felicidade', 'Tristeza', 'Surpresa', 'Neutra']  # 7 emoções do FER+
    expressao_nome = expressoes[expressao]  # Pega o nome da expressão detectada
    
    return expressao_nome

# Função principal para capturar vídeo, detectar rostos e reconhecer expressões faciais
def main():
    cameras = listar_cameras()
    
    if len(cameras) == 0:
        print("Nenhuma câmera encontrada.")
        return
    
    print("Câmeras disponíveis:")
    for i, cam in enumerate(cameras):
        print(f"{i}: Câmera {cam}")
    
    escolha = int(input("Escolha uma câmera (número): "))
    if escolha < 0 or escolha >= len(cameras):
        print("Opção inválida.")
        return

    # Iniciando a captura de vídeo
    cap = cv2.VideoCapture(cameras[escolha])

    # Carregar o modelo FER+
    modelo = carregar_modelo()

    # Inicializando o MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)  # Confiança entre 0 e 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converter a imagem para RGB (MediaPipe exige)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_detection.process(frame_rgb)
        
        if resultados.detections:
            for detection in resultados.detections:
                # Obter a caixa delimitadora (bounding box) da face detectada
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                
                # Desenhar a caixa delimitadora
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Criar um dicionário com a localização da face
                face_landmarks = {'bbox': (x, y, w, h)}
                
                # Processar a expressão facial
                expressao = processar_expressao(frame, modelo, face_landmarks)
                
                # Colocar o nome da expressão no vídeo
                cv2.putText(frame, expressao, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Exibir o vídeo
        cv2.imshow('Detecção de Expressões Faciais', frame)
        
        # Fechar o loop ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
