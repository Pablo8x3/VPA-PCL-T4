import cv2 
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time 

# --- I. CONFIGURACIÓN Y CALIBRACIÓN DE ESCALA ---

MODELO_YOLO = "yolov8n.pt"
VIDEO_PATH = "videos/lu_2.mp4"
CONFIDENCE_THRESHOLD = 0.40
MIN_TRACK_HISTORY = 2       
MIN_SPEED_KMH = 2.0         
FRAMES_REQUIRED_FOR_PERSISTENCE = 10 

# Definiciones de la línea de calibración (Eje X)
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
    FPS = 30
FRAME_TIME_SEC = 1.0 / FPS

# Almacenamiento
track_history = defaultdict(list) 
direction_persistence = defaultdict(int) 
current_speeds = {} 

# Configuración de ventana redimensionable para la VISUALIZACIÓN
cv2.namedWindow("Video Frame", cv2.WINDOW_NORMAL) 


# --- II. INICIALIZACIÓN DE VIDEO WRITER (EXPORTACIÓN AL 100%) ---

OUTPUT_FILE = 'video_velocidad_analizado.mp4' 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

# Obtener dimensiones originales para la inicialización
ret, temp_frame = video.read()
if not ret:
    video.release()
    raise Exception("No se pudo leer el primer frame para inicializar el VideoWriter.")
video.set(cv2.CAP_PROP_POS_FRAMES, 0) 

# Dimensiones del frame combinado (VIDEO ORIGINAL + BARRA DE ESTADÍSTICAS)
STATS_BAR_HEIGHT = 80
original_height, original_width, _ = temp_frame.shape
combined_width = original_width
combined_height = original_height + STATS_BAR_HEIGHT

# Inicializar VideoWriter con el tamaño 100%
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (combined_width, combined_height))

# Factor de reducción para la visualización en pantalla (75%)
DISPLAY_SCALE_FACTOR = 0.75 

# --- III. BUCLE PRINCIPAL DE PROCESAMIENTO ---

while True:
    ret, frame = video.read()
    if not ret:
        print("Fin del video.")
        break

    # 1. Detección y Tracking
    results = modelo.track(frame, persist=True, classes=[2], conf=CONFIDENCE_THRESHOLD, tracker="botsort.yaml")

    display_frame = frame.copy()
    cv2.line(display_frame, tuple(LINEA_P1), tuple(LINEA_P2), (255, 0, 0), 2)
    
    current_speeds.clear()
    current_track_ids = set() 

    if results[0].boxes.id is not None:
        
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        current_track_ids = set(track_ids)

        for box_xyxy, box_xywh, track_id in zip(boxes_xyxy, boxes_xywh, track_ids):
            
            x1, y1, x2, y2 = map(int, box_xyxy)
            centroid_x = int(box_xywh[0]) 
            
            track_history[track_id].append(centroid_x)
            
            if len(track_history[track_id]) > MIN_TRACK_HISTORY:
                track_history[track_id] = track_history[track_id][-MIN_TRACK_HISTORY:] 
            
            speed_kmh = 0.0
            
            if len(track_history[track_id]) == MIN_TRACK_HISTORY:
                
                x_end = track_history[track_id][1]
                x_start = track_history[track_id][0]
                delta_x_pixels_signed = x_end - x_start 
                
                if delta_x_pixels_signed > 0: 
                    
                    direction_persistence[track_id] += 1
                    
                    if direction_persistence[track_id] >= FRAMES_REQUIRED_FOR_PERSISTENCE:
                        
                        delta_x_pixels = abs(delta_x_pixels_signed)
                        delta_t_sec = FRAME_TIME_SEC 
                        
                        speed_px_sec = delta_x_pixels / delta_t_sec
                        speed_m_sec = speed_px_sec / PIXELS_PER_METER
                        speed_kmh = speed_m_sec * 3.6
                        
                        if speed_kmh >= MIN_SPEED_KMH:
                            current_speeds[track_id] = speed_kmh
                            
                else: 
                    direction_persistence[track_id] = 0
            
            # VISUALIZACIÓN BB
            color = (0, 255, 255) if track_id in current_speeds else (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1) 
            
            text_id = f"ID: {track_id}"
            cv2.putText(display_frame, text_id, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 4. VISUALIZACIÓN DE ESTADÍSTICAS EN LA PARTE INFERIOR

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


    # --- V. ESCRITURA Y MUESTRA ---
    
    # 1. GUARDAR (Escribir el frame AL 100%)
    out.write(final_frame) 

    # 2. VISUALIZAR (Redimensionar para la pantalla al 75%)
    
    new_width = int(combined_width * DISPLAY_SCALE_FACTOR)
    new_height = int(combined_height * DISPLAY_SCALE_FACTOR)

    resized_frame = cv2.resize(final_frame, (new_width, new_height), 
                               interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Video Frame", resized_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# --- VI. CIERRE Y LIBERACIÓN DE RECURSOS ---
out.release()
video.release()
cv2.destroyAllWindows()