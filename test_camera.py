import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("No se pudo abrir la camara")
    exit()

print("Camara abierta correctamente")

while True:
    success, frame = camera.read()

    if not success:
        print("No se pudo leer un frame")
        break

    cv2.imshow("Test camara", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()