import cv2
import mediapipe as mp
import time

from gesture_control.cursor.cursor_controller import CursorController


MODEL_PATH = "hand_landmarker.task"


# configuracion de mediapipe
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


# se carga el modelo que detecta la mano
landmarker = HandLandmarker.create_from_options(options)


# se crea el controlador del mouse
cursor = CursorController()


# se inicializa la camara
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: No se pudo abrir la camara.")
    exit()


# se usa para generar el timestamp que necesita mediapipe
start_time = time.time()


print("GestureControl iniciado")
print("Mueve tu dedo indice para controlar el mouse")
print("Presiona ESC para salir")


while True:

    # se obtiene un frame de la camara
    success, frame = camera.read()

    if not success:
        print("ERROR leyendo la camara")
        break


    # se voltea la imagen para que funcione como espejo
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # mediapipe trabaja con RGB y opencv entrega BGR
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # se convierte el frame al formato que usa mediapipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # mediapipe necesita un tiempo creciente cuando se usa en modo video
    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )


    # se procesa el frame para buscar una mano
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    if result.hand_landmarks:

        # se toma la primera mano detectada
        hand = result.hand_landmarks[0]


        # el landmark 8 corresponde a la punta del dedo indice
        index_tip = hand[8]

        finger_x = index_tip.x
        finger_y = index_tip.y


        # se mueve el mouse usando la posicion del dedo
        cursor.move(
            finger_x,
            finger_y
        )


        # se dibujan los puntos de la mano
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


        # se dibuja un punto mas grande en la punta del indice
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


    # se muestra la camara
    cv2.imshow(
        "GestureControl - ESC para salir",
        frame
    )


    # se cierra el programa al presionar ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


# se liberan los recursos al cerrar
camera.release()
cv2.destroyAllWindows()
landmarker.close()