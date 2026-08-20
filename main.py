import cv2
import mediapipe as mp
import pyautogui
import time

#elimina las pausas automaticas de pyautogui para evitar retrasos en el movimiento del mouse
pyautogui.PAUSE = 0

MODEL_PATH = "hand_landmarker.task"

# Tamaño del monitor
screen_width, screen_height = pyautogui.size()


# ==========================
# MEDIAPIPE
# ==========================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)


landmarker = HandLandmarker.create_from_options(options)


# ==========================
# CAMARA
# ==========================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: No se pudo abrir la cámara.")
    exit()


start_time = time.time()


print("GestureControl iniciado")
print("Mueve tu dedo indice para controlar el mouse")
print("Presiona ESC para salir")


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR leyendo la cámara")
        break


    # Efecto espejo
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # OpenCV = BGR
    # MediaPipe = RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Convertir a formato MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # MediaPipe VIDEO necesita timestamp creciente
    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )


    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ==========================
    # MANO DETECTADA
    # ==========================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        # Landmark 8 = punta del indice
        index_tip = hand[8]


        finger_x = index_tip.x
        finger_y = index_tip.y


        # ==========================
        # POSICION DEL MOUSE
        # ==========================

        mouse_x = int(
            finger_x * screen_width
        )

        mouse_y = int(
            finger_y * screen_height
        )


        # Evitar coordenadas fuera del monitor
        mouse_x = max(
            0,
            min(screen_width - 1, mouse_x)
        )

        mouse_y = max(
            0,
            min(screen_height - 1, mouse_y)
        )


        # Mover mouse
        pyautogui.moveTo(
            mouse_x,
            mouse_y,
            duration=0
        )


        # ==========================
        # DIBUJAR LANDMARKS
        # ==========================

        for landmark in hand:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )


            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


        # Punto rojo en el indice
        index_x = int(
            finger_x * width
        )

        index_y = int(
            finger_y * height
        )


        cv2.circle(
            frame,
            (index_x, index_y),
            12,
            (0, 0, 255),
            -1
        )


        cv2.putText(
            frame,
            "Mano detectada",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


    else:

        cv2.putText(
            frame,
            "Buscando mano...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # Mostrar cámara
    cv2.imshow(
        "GestureControl - ESC para salir",
        frame
    )


    # ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ==========================
# CERRAR
# ==========================

camera.release()

cv2.destroyAllWindows()

landmarker.close()