import cv2
import mediapipe as mp
import random
import time
import math

# Inicializa MediaPipe e webcam
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(1)

# Função para distância
def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# Função para mostrar a tela de escolha de dedo
def choose_finger():
    print("Escolha o dedo para clicar na bolinha:")
    print("1. Polegar")
    print("2. Indicador")
    print("3. Médio")
    print("4. Anelar")
    print("5. Mindinho")

    while True:
        try:
            choice = int(input("Digite o número correspondente ao dedo: "))
            if choice in [1, 2, 3, 4, 5]:
                return choice
            else:
                print("Escolha inválida. Por favor, digite um número de 1 a 5.")
        except ValueError:
            print("Escolha inválida. Por favor, digite um número de 1 a 5.")

# Solicita ao usuário escolher o dedo
finger_choice = choose_finger()

# Inicia o jogo
score = 0
target_radius = 40
target_x = random.randint(target_radius, 640 - target_radius)
target_y = random.randint(target_radius, 480 - target_radius)

touch_start_time = None
required_hold_time = 0.5  # segundos

# Verifica se a webcam foi aberta corretamente
if not cap.isOpened():
    print("❌ Não foi possível abrir a webcam.")
    exit()

# Lê primeiro frame
success, frame = cap.read()
if not success:
    print("❌ Não foi possível capturar o primeiro frame.")
    exit()

height, width, _ = frame.shape
print(f"📸 Resolução da webcam: {width}x{height}")

# Ativa modo tela cheia
cv2.namedWindow("Jogo - Toque com o dedo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Jogo - Toque com o dedo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Loop do jogo
while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    mindinho_tocando = False
    progress = 0

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Escolhe o dedo conforme a escolha feita
            if finger_choice == 1:
                lm = handLms.landmark[4]  # Polegar
            elif finger_choice == 2:
                lm = handLms.landmark[8]  # Indicador
            elif finger_choice == 3:
                lm = handLms.landmark[12]  # Médio
            elif finger_choice == 4:
                lm = handLms.landmark[16]  # Anelar
            elif finger_choice == 5:
                lm = handLms.landmark[20]  # Mindinho

            x = int(lm.x * width)
            y = int(lm.y * height)

            # Desenhar ponto do dedo escolhido
            cv2.circle(frame, (x, y), 15, (255, 0, 0), cv2.FILLED)

            # Verifica se o dedo está tocando o alvo
            if distance(x, y, target_x, target_y) < target_radius:
                mindinho_tocando = True
            else:
                touch_start_time = None

    # Lógica do clique contínuo
    if mindinho_tocando:
        if touch_start_time is None:
            touch_start_time = time.time()
        else:
            duration = time.time() - touch_start_time
            progress = duration / required_hold_time

            # Quando segura por tempo suficiente
            if progress >= 1.0:
                score += 1
                target_x = random.randint(target_radius, width - target_radius)
                target_y = random.randint(target_radius, height - target_radius)
                touch_start_time = None
                progress = 0
    else:
        progress = 0
        touch_start_time = None

    # Desenha o alvo com feedback
    if progress > 0:
        cv2.circle(frame, (target_x, target_y), target_radius, (0, 255, 0), -1)
    else:
        cv2.circle(frame, (target_x, target_y), target_radius, (0, 255, 255), -1)

    # Desenha a barra circular de progresso
    if progress > 0:
        angle = int(progress * 360)
        cv2.ellipse(frame, (target_x, target_y), (target_radius + 10, target_radius + 10),
                    0, 0, angle, (0, 255, 0), 4)

    # Mostrar pontuação
    cv2.putText(frame, f'Score: {score}', (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    # Mostrar na tela
    cv2.imshow("Jogo - Toque com o dedo", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc para sair
        break

# Finalizar
cap.release()
cv2.destroyAllWindows()
