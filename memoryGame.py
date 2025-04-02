import cv2
import mediapipe as mp
import random
import time
import numpy as np

# Configurações
ROWS, COLS = 2, 3
CARD_WIDTH, CARD_HEIGHT = 160, 160
SPACING = 20
HOLD_TIME = 2
SHOW_TIME = 5
WRONG_PAIR_SHOW_TIME = 2

# Gera 3 pares de cores
def generate_images():
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    images = []
    for color in colors:
        img = np.zeros((CARD_HEIGHT, CARD_WIDTH, 3), dtype=np.uint8)
        img[:] = color
        images.append(img)
    return images * 2

def create_board():
    imgs = generate_images()
    random.shuffle(imgs)
    board = []
    idx = 0
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            row.append({
                "img": imgs[idx],
                "revealed": False,
                "matched": False
            })
            idx += 1
        board.append(row)
    return board

def point_in_rect(px, py, x, y, w, h):
    return x <= px <= x + w and y <= py <= y + h

# Inicialização
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

success, frame = cap.read()
height, width, _ = frame.shape

# Centralização do tabuleiro
total_width = COLS * CARD_WIDTH + (COLS - 1) * SPACING
total_height = ROWS * CARD_HEIGHT + (ROWS - 1) * SPACING
offset_x = (width - total_width) // 2
offset_y = (height - total_height) // 2

def get_card_pos(row, col):
    x = col * (CARD_WIDTH + SPACING) + offset_x
    y = row * (CARD_HEIGHT + SPACING) + offset_y
    return x, y

# Criação do jogo
board = create_board()
start_time = time.time()
score = 0
selected_cards = []
touch_start_time = None
current_target = None
pending_hide = None
pending_start_time = None
game_over = False

# Tela fullscreen
cv2.namedWindow("Jogo da Memória", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Jogo da Memória", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Loop principal
while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    indicador_x, indicador_y = None, None
    dedo_tocando = None
    now = time.time()
    show_images = now - start_time < SHOW_TIME

    if score == 3:
        game_over = True

    if game_over:
        frame[:] = (0, 0, 0)
        cv2.putText(frame, "Você venceu!", (width // 2 - 300, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
        cv2.putText(frame, "Pressione ESC para sair", (width // 2 - 300, height // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
        cv2.imshow("Jogo da Memória", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Desenha cartas
    for r in range(ROWS):
        for c in range(COLS):
            x, y = get_card_pos(r, c)
            card = board[r][c]
            show = card["revealed"] or card["matched"] or show_images

            card_img = card["img"] if show else np.zeros((CARD_HEIGHT, CARD_WIDTH, 3), dtype=np.uint8) + 50
            resized_card = cv2.resize(card_img, (CARD_WIDTH, CARD_HEIGHT))
            if y + CARD_HEIGHT <= frame.shape[0] and x + CARD_WIDTH <= frame.shape[1]:
                frame[y:y+CARD_HEIGHT, x:x+CARD_WIDTH] = resized_card

    # Detecta o dedo indicador (landmark 8)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark[8]
            indicador_x = int(lm.x * width)
            indicador_y = int(lm.y * height)

    # Verifica se está sobre alguma carta
    if indicador_x and indicador_y:
        for r in range(ROWS):
            for c in range(COLS):
                x, y = get_card_pos(r, c)
                card = board[r][c]

                if point_in_rect(indicador_x, indicador_y, x, y, CARD_WIDTH, CARD_HEIGHT) and not card["revealed"] and not card["matched"] and not show_images:
                    dedo_tocando = (r, c)

    # Lógica de clique
    if dedo_tocando and not pending_hide:
        if touch_start_time is None or current_target != dedo_tocando:
            touch_start_time = time.time()
            current_target = dedo_tocando
        else:
            elapsed = time.time() - touch_start_time
            progress = min(elapsed / HOLD_TIME, 1.0)

            # Animação do clique (círculo azul preenchendo)
            cv2.ellipse(
                frame,
                (indicador_x, indicador_y),
                (30, 30),
                -90, 0, int(progress * 360),
                (255, 0, 0), 5
            )

            if progress >= 1.0:
                r, c = dedo_tocando
                board[r][c]["revealed"] = True
                selected_cards.append((r, c))
                touch_start_time = None
                current_target = None

                if len(selected_cards) == 2:
                    r1, c1 = selected_cards[0]
                    r2, c2 = selected_cards[1]
                    card1 = board[r1][c1]
                    card2 = board[r2][c2]

                    if np.array_equal(card1["img"], card2["img"]):
                        card1["matched"] = True
                        card2["matched"] = True
                        score += 1
                        selected_cards = []
                    else:
                        pending_hide = [(r1, c1), (r2, c2)]
                        pending_start_time = time.time()
    else:
        touch_start_time = None
        current_target = None

    # Oculta cartas erradas após tempo
    if pending_hide and time.time() - pending_start_time >= WRONG_PAIR_SHOW_TIME:
        for r, c in pending_hide:
            board[r][c]["revealed"] = False
        pending_hide = None
        selected_cards = []

    # Desenha dedo indicador
    if indicador_x and indicador_y:
        cv2.circle(frame, (indicador_x, indicador_y), 12, (255, 0, 0), -1)

    # Pontuação
    cv2.putText(frame, f"Score: {score}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow("Jogo da Memória", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
