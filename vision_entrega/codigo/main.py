import cv2 
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time 


MODELO_YOLO = "yolov8n.pt"
VIDEO_PATH = "videos/lu_2.mp4"
CONFIDENCE_THRESHOLD = 0.40
MIN_TRACK_HISTORY = 2       # Usamos los últimos 2 puntos para estabilizar la velocidad
MIN_SPEED_KMH = 2.0         # Mínima velocidad para mostrar (hay coches parados y no quiero que cuenten)
FRAMES_REQUIRED_FOR_PERSISTENCE = 10 # Mínimos frames consecutivos en la misma dirección

# Definiciones de la línea de calibración (Eje X)
# [Punto 1 Derecha], [Punto 2 Izquierda], ...
rdi = np.array([(1880, 565), (43, 565), (1266, 137),(641, 143) ], dtype=np.int32)
LINEA_P1 = rdi[0]
LINEA_P2 = rdi[1]
LONGITUD_REAL_METROS = 22.5 

# Calibración
distancia_base_pixeles = abs(LINEA_P1[0] - LINEA_P2[0])
PIXELS_PER_METER = distancia_base_pixeles / LONGITUD_REAL_METROS 


modelo = YOLO(MODELO_YOLO)
video = cv2.VideoCapture(VIDEO_PATH)

FPS = video.get(cv2.CAP_PROP_FPS) 
if FPS == 0:
    FPS = 30 # Valor por defecto si no se puede leer
FRAME_TIME_SEC = 1.0 / FPS

# Almacenamiento de posición del centroide X
track_history = defaultdict(list) 

# Almacena el contador de frames consecutivos en la dirección derecha
direction_persistence = defaultdict(int) 

current_speeds = {} 

# Configuración de ventana redimensionable; el video es 1080p y al verlo completo no cabe en pantalla por eso esto 
cv2.namedWindow("Video Frame", cv2.WINDOW_NORMAL) 

# --- II. BUCLE PRINCIPAL DE PROCESAMIENTO ---

while True:
    ret, frame = video.read()
    if not ret:
        print("Fin del video.")
        break

    # 1. Detección y Tracking
    results = modelo.track(frame, persist=True, classes=[2], conf=CONFIDENCE_THRESHOLD, tracker="botsort.yaml")

    display_frame = frame.copy()
    
    # Dibujar la línea de referencia horizontal
    cv2.line(display_frame, tuple(LINEA_P1), tuple(LINEA_P2), (255, 0, 0), 2)
    
    current_speeds.clear()
    current_track_ids = set() 

    if results[0].boxes.id is not None:
        
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        current_track_ids = set(track_ids)

        # 2. Iterar sobre los resultados
        for box_xyxy, box_xywh, track_id in zip(boxes_xyxy, boxes_xywh, track_ids):
            
            x1, y1, x2, y2 = map(int, box_xyxy)
            centroid_x = int(box_xywh[0]) 
            
            # --- CÁLCULO DE VELOCIDAD Y FILTROS ---
            
            track_history[track_id].append(centroid_x)
            
            if len(track_history[track_id]) > MIN_TRACK_HISTORY:
                track_history[track_id] = track_history[track_id][-MIN_TRACK_HISTORY:] 
            
            speed_kmh = 0.0
            
            if len(track_history[track_id]) == MIN_TRACK_HISTORY:
                
                x_end = track_history[track_id][1]
                x_start = track_history[track_id][0]
                delta_x_pixels_signed = x_end - x_start 
                
                # 3. FILTRO DE DIRECCIÓN Y PERSISTENCIA (Solo derecha > 0)
                if delta_x_pixels_signed > 0: 
                    
                    direction_persistence[track_id] += 1
                    
                    if direction_persistence[track_id] >= FRAMES_REQUIRED_FOR_PERSISTENCE:
                        
                        delta_x_pixels = abs(delta_x_pixels_signed)
                        delta_t_sec = FRAME_TIME_SEC 
                        
                        # Cálculo
                        speed_px_sec = delta_x_pixels / delta_t_sec
                        speed_m_sec = speed_px_sec / PIXELS_PER_METER
                        speed_kmh = speed_m_sec * 3.6
                        
                        if speed_kmh >= MIN_SPEED_KMH:
                            current_speeds[track_id] = speed_kmh
                            
                else: 
                    # Reiniciar la persistencia si se mueve a la izquierda o está parado
                    direction_persistence[track_id] = 0
            
            # --- VISUALIZACIÓN BB (Sin texto de velocidad) ---
            color = (0, 255, 255) if track_id in current_speeds else (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1) 
            
            # Dibujar el ID sobre el BB para referencia
            text_id = f"ID: {track_id}"
            cv2.putText(display_frame, text_id, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # PPongo las velocidades en la parte inferior ya que si las dejo en la caja no se ven bine y es imposible de distinguir a lo largo de la pantalla

    frame_height, frame_width, _ = display_frame.shape
    stats_height = 80
    stats_bar = np.zeros((stats_height, frame_width, 3), np.uint8)
    
    y_offset = 30
    x_start = 20
    
    cv2.putText(stats_bar, f"Velocidades (Derecha, >{FRAMES_REQUIRED_FOR_PERSISTENCE} frames):", (x_start, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    col_index = 0
    y_offset += 30 
    
    for track_id, speed in current_speeds.items():
        text = f"ID {track_id}: {speed:.1f} km/h"
        x_pos = x_start + col_index * 250 
        
        cv2.putText(stats_bar, text, (x_pos, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        col_index += 1
        
    final_frame = cv2.vconcat([display_frame, stats_bar])


    
    SCALE_FACTOR = 1 # Reducir al 75%
    frame_height, frame_width, _ = final_frame.shape

    new_width = int(frame_width * SCALE_FACTOR)
    new_height = int(frame_height * SCALE_FACTOR)

    resized_frame = cv2.resize(final_frame, (new_width, new_height), 
                               interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Video Frame", resized_frame)

    key = cv2.waitKey(1)
    if key == 27:  # Tecla 'Esc'
        break


video.release()
cv2.destroyAllWindows()