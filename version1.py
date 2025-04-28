import cv2
import cv2.text
import mediapipe as mp
from math import dist
import time
# ----- IMPORTANTE!!! -------

miCamara = 0  # Cambiar por 1, 2, etc. si la cámara no se detecta

# ---------------------------
# Inicializar el modelo de detección de manos
# A diferencia de nuestro modelo del OCR, éste ya está entrenado!
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7,
                      )
mpDraw = mp.solutions.drawing_utils
# Anotamos la coordenada de cada dedo según la imagen
dedos = {
    "indice":[6,8],
    "anular":[10,12],
    "mayor":[14,16],
    "menyque":[18,20],
    "pulgar":[3,4]
}
def coord_x(marcador):  
    return float(str(results.multi_hand_landmarks[-1].landmark[int(marcador)]).split('\n')[0].split(" ")[1])
def coord_y(marcador): 
    return float(str(results.multi_hand_landmarks[-1].landmark[int(marcador)]).split('\n')[1].split(" ")[1])
def detectarDedo():   
    if results.multi_hand_landmarks is not None:
        try:
        # calculamos la distancia entre la punta del dedo y la palma. 
        # y entre la parte media del dedo y la palma.
        # si la distancia de la punta es mas corta que la del medio
        # el dedo está cerrado
            x_palma = coord_x(0) 
            y_palma = coord_y(0)
            cerrados = []
            for medio,punta in dedos.values(): 
                x_medio = coord_x(medio)
                y_medio = coord_y(medio) #coordenadas x e y de la mitad del dedo
                x_punta = coord_x(punta)
                y_punta = coord_y(punta) # coordenadas x e y de la punta del dedo
                d_medio = dist([x_palma, y_palma], [x_medio, y_medio])
                d_punta = dist([x_palma, y_palma], [x_punta, y_punta])
                if d_medio < d_punta:
                    cerrados.append(1)
                else:
                    cerrados.append(0)
            return cerrados                 
        except:
           pass
# Inicializar la cámara en el índice
cap = cv2.VideoCapture(miCamara)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convertir la imagen de BGR a RGB
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Realizar la detección
    results = hands.process(imgRGB)
    # Dibujar las detecciones
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                d = [8,12,16,20,4]
                deteccion = detectarDedo()
                # Pintamos de color verde o rojo la punta de los dedos
                if id in d:
                    if deteccion[d.index(id)] == 1:
                        cv2.circle(frame,(cx,cy),10,(0,255,0),-1) # Dedo arriba, verde
                    else:
                        cv2.circle(frame,(cx,cy),10,(0,0,255),-1) # Dedo abajo, rojo
            # Esta linea de código dibuja las marcas que unen los puntos
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS) 
    # llamamos a nuestra función para encontrar qué dedos están arriba
    deteccion = detectarDedo()
    if deteccion is not None:
        # Anotamos cuántos dedos están arriba
        cv2.putText(frame,f"dedos detectados={sum(deteccion)}",(20,30),fontFace=0,fontScale=1,color=(0,255,0))    
    
    # Mostrar la imagen
    cv2.imshow('Deteccion de mano', frame)
    # Pequeño delay entre cuadros, especialmente si tratamos con videos
    time.sleep(1/30)
    # Salir del bucle al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()