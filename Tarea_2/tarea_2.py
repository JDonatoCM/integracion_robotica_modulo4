"""
Tarea 2. Kalman filter application

Detector y Seguimiento de Caballos utilizando YOLOv8 con Trayectorias

Este script implementa un sistema de detección y seguimiento de caballos en video utilizando
el modelo YOLOv8 de Ultralytics. El programa procesa un
archivo de video para identificar caballos, dibuja cajas delimitadoras alrededor
de ellos, asigna identificadores únicos y muestra la trayectoria que sigue cada caballo.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import random

def load_model(model_path):
    """
    Carga el modelo YOLOv8 desde la ruta especificada.
    
    Args:
        model_path (str): Ruta al archivo del modelo pre-entrenado YOLOv8.
        
    Returns:
        YOLO: Instancia del modelo YOLO cargado.
        
    Raises:
        Exception: Si hay algún error al cargar el modelo.
    """
    try:
        model = YOLO(model_path)
        print(f"Modelo YOLOv8 cargado exitosamente desde: {model_path}")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        raise

def open_video(filepath):
    """
    Abre un archivo de video para su procesamiento.
    
    Args:
        filepath (str): Ruta al archivo de video.
        
    Returns:
        cv2.VideoCapture: Objeto VideoCapture para el procesamiento del video.
        
    Raises:
        SystemExit: Si no se puede abrir el archivo de video.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {filepath}.")
        exit()
    
    # Obtener información del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video abierto - Resolución: {width}x{height}, FPS: {fps}")
    return cap

def process_frame_with_tracking(frame, model, track_history, colors_dict, horse_ids, confidence_threshold=0.25):
    """
    Procesa un fotograma para detectar y seguir caballos usando YOLOv8 con tracking.
    
    Args:
        frame (numpy.ndarray): Fotograma del video a procesar.
        model (YOLO): Modelo YOLOv8 cargado.
        track_history (dict): Diccionario que almacena el historial de posiciones para cada ID.
        colors_dict (dict): Diccionario que asigna un color a cada ID de caballo.
        horse_ids (dict): Diccionario que asigna un nombre (Caballo N) a cada ID de tracking.
        confidence_threshold (float): Umbral de confianza para filtrar detecciones.
                                    Default es 0.25.
    
    Returns:
        numpy.ndarray: Fotograma con las detecciones y trayectorias dibujadas.
        int: Número de caballos detectados en este fotograma.
    """
    # Ejecutar la detección y tracking en el fotograma actual
    # No filtrar por clase aquí para asegurarnos de detectar todos los objetos
    results = model.track(frame, persist=True, conf=confidence_threshold)
    
    # Contador de caballos detectados
    horse_count = 0
    
    # Si hay resultados
    if results and len(results) > 0:
        # Extraer las detecciones del primer resultado
        detections = results[0]
        
        # Verificar si hay cajas detectadas
        if detections.boxes and len(detections.boxes) > 0:
            # Obtener las cajas e IDs de tracking (si están disponibles)
            boxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            confs = detections.boxes.conf.cpu().numpy()
            
            # Verificar si hay IDs de tracking
            has_tracking = hasattr(detections.boxes, 'id') and detections.boxes.id is not None
            if has_tracking:
                track_ids = detections.boxes.id.int().cpu().numpy()
            
            # Procesar cada detección
            for i in range(len(boxes)):
                class_id = int(classes[i])
                class_name = detections.names[class_id]
                
                # Filtrar solo para caballos (clase 13 en COCO o nombre 'horse')
                if class_name == 'horse' or class_id == 13:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    confidence = confs[i]
                    
                    # Continuar solo si la confianza es mayor que el umbral
                    if confidence < confidence_threshold:
                        continue
                    
                    # Calcular el centro para la trayectoria
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    center_point = (center_x, center_y)
                    
                    # Asignar ID para tracking
                    if has_tracking:
                        track_id = int(track_ids[i])
                    else:
                        # Si no hay tracking, usar un ID basado en la posición
                        track_id = i
                    
                    # Inicializar tracking para nuevo ID
                    if track_id not in track_history:
                        track_history[track_id] = []
                        # Generar color aleatorio para nuevo ID
                        colors_dict[track_id] = (
                            random.randint(50, 255),
                            random.randint(50, 255),
                            random.randint(50, 255)
                        )
                        # Asignar nombre a nuevo caballo
                        horse_ids[track_id] = f"Caballo {len(horse_ids) + 1}"
                    
                    # Guardar posición actual para la trayectoria
                    track_history[track_id].append(center_point)
                    
                    # Obtener color y nombre para este caballo
                    color = colors_dict[track_id]
                    horse_name = horse_ids[track_id]
                    
                    # Dibujar caja delimitadora
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Dibujar etiqueta con nombre
                    label_size, _ = cv2.getTextSize(horse_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, horse_name, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
                    # Limitar historial de trayectoria para no sobrecargar memoria
                    max_history = 50
                    if len(track_history[track_id]) > max_history:
                        track_history[track_id] = track_history[track_id][-max_history:]
                    
                    # Dibujar trayectoria
                    points = track_history[track_id]
                    if len(points) >= 2:
                        for j in range(1, len(points)):
                            # Dibujar línea entre puntos consecutivos
                            cv2.line(frame, points[j-1], points[j], color, 2)
                    
                    # Dibujar punto actual
                    cv2.circle(frame, center_point, 5, color, -1)
                    
                    # Incrementar contador de caballos
                    horse_count += 1
    
    # Añadir contador de caballos al fotograma
    cv2.putText(frame, f'Caballos detectados: {horse_count}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame, horse_count

def main():
    """
    Función principal que ejecuta el sistema de detección y seguimiento de caballos.
    
    Esta función inicializa el modelo, abre el video, procesa cada fotograma
    para detectar y seguir caballos, y muestra los resultados en tiempo real.
    """
    # Configuración del modelo y video
    model_path = 'yolov8m.pt'  # Modelo de tamaño medio: balance entre precisión y velocidad
    video_path = 'video_caballos_3.mp4' # Ruta al archivo de video (ubicar en la misma carpeta)
    confidence_threshold = 0.25  # Umbral de confianza para detecciones
    
    # Cargar modelo YOLOv8
    print("Inicializando detector de caballos...")
    model = load_model(model_path)
    
    # Abrir video
    print(f"Abriendo video: {video_path}")
    cap = open_video(video_path)
    
    # Diccionarios y variables para el tracking y trayectorias
    track_history = {}  # Diccionario para almacenar historial de posiciones de cada ID
    colors_dict = {}    # Diccionario para asignar colores a cada ID
    horse_ids = {}      # Diccionario para asignar nombres (Caballo 1, Caballo 2, etc.) a cada ID
    
    # Variables para estadísticas
    total_frames = 0
    total_horses_detected = 0
    
    print("Comenzando procesamiento del video. Presiona 'q' para salir.")
    
    # Bucle principal de procesamiento de video
    while True:
        # Leer el siguiente fotograma
        ret, frame = cap.read()
        
        # Salir si no hay más fotogramas
        if not ret:
            print("Fin del video o error en la lectura.")
            break
        
        # Incrementar contador de fotogramas
        total_frames += 1
        
        # Procesar fotograma para detectar y seguir caballos
        processed_frame, horses_in_frame = process_frame_with_tracking(
            frame, model, track_history, colors_dict, horse_ids, confidence_threshold)
        
        # Actualizar estadísticas
        total_horses_detected += horses_in_frame
        
        # Mostrar fotograma con detecciones y trayectorias
        cv2.imshow("Detección y Seguimiento de Caballos - YOLOv8", processed_frame)
        
        # Verificar si el usuario quiere salir (tecla 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Procesamiento interrumpido por el usuario.")
            break
    
    # Mostrar estadísticas finales
    print(f"Procesamiento completado.")
    print(f"Total de fotogramas procesados: {total_frames}")
    if total_frames > 0:
        print(f"Promedio de caballos por fotograma: {total_horses_detected/total_frames:.2f}")
    print(f"Caballos únicos detectados: {len(horse_ids)}")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Recursos liberados. Programa finalizado correctamente.")

if __name__ == "__main__":
    """
    Punto de entrada del script cuando se ejecuta directamente.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")
        import traceback
        traceback.print_exc()  # Imprimir el traceback completo para depuración
