import cv2
import mediapipe as mp
from math import dist
import time
import numpy as np
import pyautogui

# Configuración inicial
miCamara = 0
pyautogui.FAILSAFE = False  # Desactivar protección para mover mouse a esquinas

# Inicializar modelos de MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Landmarks de los dedos
dedos = {
    "indice": [6, 8],
    "anular": [10, 12],
    "mayor": [14, 16],
    "menyque": [18, 20],
    "pulgar": [3, 4]
}

# Gestos predefinidos
GESTOS = {
    (1, 0, 0, 0, 0): "Pulgar arriba 👍",
    (0, 1, 0, 0, 0): "Señalar 👆",
    (0, 1, 1, 0, 0): "Victoria ✌️",
    (1, 1, 1, 1, 1): "Mano abierta 🖐️",
    (0, 0, 0, 0, 0): "Puño ✊"
}

# Configuración de la interfaz
COLORES = {
    'fondo': (50, 50, 50),
    'texto': (255, 255, 255),
    'dedo_arriba': (0, 255, 0),
    'dedo_abajo': (0, 0, 255),
    'gesto_detectado': (255, 255, 0)
}

# Tamaño de la pantalla para control del mouse
pantalla_ancho, pantalla_alto = pyautogui.size()

def coord_x(marcador):  
    return float(str(results.multi_hand_landmarks[-1].landmark[int(marcador)]).split('\n')[0].split(" ")[1])

def coord_y(marcador): 
    return float(str(results.multi_hand_landmarks[-1].landmark[int(marcador)]).split('\n')[1].split(" ")[1])

def detectarDedo():   
    if results.multi_hand_landmarks is not None:
        try:
            x_palma = coord_x(0) 
            y_palma = coord_y(0)
            cerrados = []
            for medio, punta in dedos.values(): 
                x_medio = coord_x(medio)
                y_medio = coord_y(medio)
                x_punta = coord_x(punta)
                y_punta = coord_y(punta)
                d_medio = dist([x_palma, y_palma], [x_medio, y_medio])
                d_punta = dist([x_palma, y_palma], [x_punta, y_punta])
                cerrados.append(1 if d_medio < d_punta else 0)
            return cerrados                 
        except:
            return None

def reconocer_gesto(estado_dedos):
    if estado_dedos is None:
        return None
    estado_tuple = tuple(estado_dedos)
    return GESTOS.get(estado_tuple, "Gestos no reconocido")

def control_mouse(estado_dedos, frame):
    if estado_dedos is None or sum(estado_dedos) != 1 or estado_dedos[1] != 1:
        return
    
    # Solo funciona cuando se señala con el índice (gesto 👆)
    x_punta = coord_x(8)
    y_punta = coord_y(8)
    
    # Convertir coordenadas relativas a posición absoluta en pantalla
    mouse_x = np.interp(x_punta, [0.1, 0.9], [0, pantalla_ancho])
    mouse_y = np.interp(y_punta, [0.1, 0.9], [0, pantalla_alto])
    
    pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)
    
    # Hacer clic si el pulgar también está levantado
    if estado_dedos[0] == 1:
        pyautogui.click()
        cv2.putText(frame, "CLIC", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

def dibujar_interfaz(frame, estado_dedos, gesto):
    # Panel de información
    cv2.rectangle(frame, (0, 0), (300, 150), COLORES['fondo'], -1)
    
    # Contador de dedos
    if estado_dedos is not None:
        cv2.putText(frame, f"Dedos levantados: {sum(estado_dedos)}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORES['texto'], 2)
        
        # Estado individual de cada dedo
        for i, (nombre, _) in enumerate(dedos.items()):
            color = COLORES['dedo_arriba'] if estado_dedos[i] == 1 else COLORES['dedo_abajo']
            cv2.putText(frame, f"{nombre}: {'↑' if estado_dedos[i] == 1 else '↓'}", (20, 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Gestos reconocidos
    if gesto:
        cv2.putText(frame, gesto, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORES['gesto_detectado'], 2)

# Inicializar cámara
cap = cv2.VideoCapture(miCamara)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Bucle principal
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Voltear la imagen para efecto espejo
    frame = cv2.flip(frame, 1)
    
    # Detección de manos
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Dibujar landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Dibujar puntos de los dedos
                if id in [8, 12, 16, 20, 4]:  # Puntas de los dedos
                    estado_dedos = detectarDedo()
                    if estado_dedos and len(estado_dedos) == 5:
                        dedo_idx = [8, 12, 16, 20, 4].index(id)
                        color = COLORES['dedo_arriba'] if estado_dedos[dedo_idx] == 1 else COLORES['dedo_abajo']
                        cv2.circle(frame, (cx, cy), 10, color, -1)
            
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    
    # Detección de gestos
    estado_dedos = detectarDedo()
    gesto = reconocer_gesto(estado_dedos) if estado_dedos else None
    
    # Control de mouse
    control_mouse(estado_dedos, frame)
    
    # Interfaz mejorada
    dibujar_interfaz(frame, estado_dedos, gesto)
    
    # Mostrar frame
    cv2.imshow('Control por Gestos Avanzado', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()