import cv2
import math
from ultralytics import YOLO

# CARGANDO EL MODELO A UTILIZAR
model = YOLO("./best.pt")
# REDIMENSIONANDO LA IMAGEN CAPTURADA
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# CLASES A DETECTAR
classNames =  ["SIN_MASCARILLA", "CON_MASCARILLA", "MARCARILLA_MAL_COLOCADA"]

while True:
    # CAPTURANDO IMAGEN DE LA WEBCAM
    success, img = cap.read() 
    # DETECTANDO OBJETOS EN LA IMAGEN
    results = model(img, stream=True) 

    # DIBUJANDO EL CUADRO DELIMITADOR Y ETIQUETANDO LOS OBJETOS DETECTADOS
    for r in results:
        marcos = r.boxes
        for marco in marcos:
            # COORDENADAS DEL BOUNDING BOX 
            x1, y1, x2, y2 = marco.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # CONVERTIR A ENTEROS

            # EXTRAYENDO EL BOUNDING BOX Y DIBUJANDO EL MARCO DELIMITADOR
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # FIABILIDAD DE LA PREDICCION
            confidence = math.ceil((marco.conf[0]*100))/100
            print("Confidence --->",confidence)

            # AGISNACION DE CLASES
            cls = int(marco.cls[0])
            try:
                name_box = classNames[cls]
                print("Class name -->", name_box)
            except IndexError:
                name_box = "Error de reconocimiento"
                print("Error de clase, no se identifica. Reentrena mejor tu modelo.")

            # ETIQUETANDO LOS OBJETOS DETECTADOS
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            # DIBUJANDO EL TEXTO
            cv2.putText(img, name_box, org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()