import cv2
import torch

def main():
    # Carga el modelo YOLOv7 preentrenado desde torch.hub
    model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True)

    # Inicializa la cámara (0 para cámara por defecto)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Realiza la detección
        results = model(frame)

        # Dibuja los resultados en la imagen
        frame = results.render()[0]

        # Muestra la imagen con detecciones
        cv2.imshow('YOLOv7 Real-Time Detection', frame)

        # Salir si presionas 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
