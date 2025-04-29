"""
Aplicación de Procesamiento de Señales de Audio.

Esta aplicación permite cargar, procesar, visualizar y guardar archivos de audio.
Incluye funcionalidades para aplicar diferentes tipos de filtros y transformadas
a las señales de audio, con una interfaz gráfica mejorada y visualización avanzada.
"""

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, 
    QFileDialog, QLabel, QComboBox, QSlider, QGroupBox, QHBoxLayout,
    QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QSound
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MainWindow(QMainWindow):
    """
    Ventana principal de la aplicación para procesamiento de señales de audio.
    
    Esta clase proporciona una interfaz gráfica para cargar archivos de audio,
    aplicar filtros, visualizar y guardar los resultados procesados.
    """
    
    def __init__(self):
        """
        Constructor de la clase.
        
        Inicializa la ventana principal y configura la interfaz de usuario.
        También inicializa las variables de instancia para almacenar los datos de audio.
        """
        super().__init__()
        self.setMinimumSize(1000, 800)
        self.setWindowTitle('HMI para procesamiento de señales')
        
        # Inicialización de variables para almacenar datos de audio
        self.audio_data = None
        self.sample_rate = None
        self.processed_audio = None
        self.fft_original = None
        self.fft_processed = None
        
        # Variables para controlar los límites de los ejes
        self.y_limits_signal = None
        self.y_limits_fft = None
        
        # Configurar la interfaz de usuario
        self.init_ui()
        
    def init_ui(self):
        """
        Configura todos los elementos de la interfaz de usuario.
        
        Crea y organiza los widgets de la interfaz, incluyendo botones,
        sliders, combos y áreas de visualización con un diseño mejorado.
        """
        # Crear layout principal
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Sección superior para controles
        top_section = QVBoxLayout()
        
        # Botón para cargar archivo (centrado con ancho limitado)
        load_button_layout = QHBoxLayout()
        self.load_button = QPushButton('Cargar archivo')
        self.load_button.setFixedWidth(200)
        self.load_button.clicked.connect(self.load_file)
        load_button_layout.addStretch()
        load_button_layout.addWidget(self.load_button)
        load_button_layout.addStretch()
        top_section.addLayout(load_button_layout)
        
        # Grupo de configuración del filtro
        filter_group = self.create_filter_group()
        top_section.addWidget(filter_group)
        
        # Botones de acción (en dos filas)
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setSpacing(10)
        
        # Botón unificado para aplicar filtro
        self.apply_filter_button = QPushButton('Procesar Señal')
        self.apply_filter_button.setFixedWidth(200)
        self.apply_filter_button.clicked.connect(self.process_signal)
        action_buttons_layout.addStretch()
        action_buttons_layout.addWidget(self.apply_filter_button)
        action_buttons_layout.addStretch()
        
        top_section.addLayout(action_buttons_layout)
        
        # Segunda fila de botones (Guardar resultado y Reproducir filtrado)
        save_play_layout = QHBoxLayout()
        save_play_layout.setSpacing(10)
        
        self.save_button = QPushButton('Guardar Resultado')
        self.save_button.setFixedWidth(200)
        self.save_button.clicked.connect(self.save_result)
        save_play_layout.addStretch()
        save_play_layout.addWidget(self.save_button)
        
        play_button = QPushButton('Reproducir Filtrado')
        play_button.setFixedWidth(200)
        play_button.clicked.connect(self.play_filtered_sound)
        save_play_layout.addWidget(play_button)
        save_play_layout.addStretch()
        
        top_section.addLayout(save_play_layout)
        
        main_layout.addLayout(top_section)
        
        # Grupo de visualización
        visualization_group = self.create_visualization_group()
        visualization_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(visualization_group, 1)  # 1 es la prioridad de estiramiento
        
        # Configurar widget central
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Aplicar estilo CSS
        self.apply_stylesheet()
        
    def create_filter_group(self):
        """
        Crea y configura el grupo de widgets para el control de filtros.
        
        Incluye dos sliders para el filtro pasa-banda, activando el segundo
        slider solo cuando se selecciona este tipo de filtro.
        
        Returns:
            QGroupBox: El grupo de widgets configurado para los controles de filtro.
        """
        filter_group = QGroupBox("Configuración del filtro")
        filter_layout = QVBoxLayout()
        
        # Combo de selección de filtro
        filter_type_layout = QHBoxLayout()
        self.filter_label = QLabel('Seleccionar filtro:')
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(['pasa-bajas', 'pasa-altas', 'pasa-banda'])
        self.filter_combo.currentTextChanged.connect(self.on_filter_type_changed)
        
        filter_type_layout.addWidget(self.filter_label)
        filter_type_layout.addWidget(self.filter_combo)
        filter_type_layout.addStretch()
        filter_layout.addLayout(filter_type_layout)
        
        # Slider para frecuencia de corte baja
        low_freq_layout = QHBoxLayout()
        self.freq_low_label = QLabel('Frecuencia de corte baja:')
        self.freq_low_slider = QSlider(Qt.Horizontal)
        self.freq_low_slider.setRange(100, 5000)
        self.freq_low_slider.setValue(1000)
        self.freq_low_slider.valueChanged.connect(self.update_freq_labels)
        self.freq_low_value = QLabel(f'{self.freq_low_slider.value()} Hz')
        
        low_freq_layout.addWidget(self.freq_low_label)
        low_freq_layout.addWidget(self.freq_low_slider)
        low_freq_layout.addWidget(self.freq_low_value)
        filter_layout.addLayout(low_freq_layout)
        
        # Slider para frecuencia de corte alta (visible solo para pasa-banda)
        high_freq_layout = QHBoxLayout()
        self.freq_high_label = QLabel('Frecuencia de corte alta:')
        self.freq_high_slider = QSlider(Qt.Horizontal)
        self.freq_high_slider.setRange(100, 10000)
        self.freq_high_slider.setValue(2000)
        self.freq_high_slider.valueChanged.connect(self.update_freq_labels)
        self.freq_high_value = QLabel(f'{self.freq_high_slider.value()} Hz')
        
        high_freq_layout.addWidget(self.freq_high_label)
        high_freq_layout.addWidget(self.freq_high_slider)
        high_freq_layout.addWidget(self.freq_high_value)
        filter_layout.addLayout(high_freq_layout)
        
        # Inicialmente ocultar el slider de frecuencia alta para filtros no pasa-banda
        self.set_high_freq_visibility(False)
        
        filter_group.setLayout(filter_layout)
        return filter_group
        
    def create_visualization_group(self):
        """
        Crea y configura el grupo de widgets para la visualización de señales.
        
        Incluye 4 gráficos: señal original, señal procesada,
        transformada de la señal original y transformada de la señal procesada.
        
        Returns:
            QGroupBox: El grupo de widgets configurado para las visualizaciones.
        """
        visualization_group = QGroupBox("Visualización de señales")
        visualization_layout = QGridLayout()
        visualization_layout.setSpacing(10)
        
        # Gráfico para señal original
        original_label = QLabel("Señal Original")
        visualization_layout.addWidget(original_label, 0, 0)
        
        self.original_signal_canvas = FigureCanvas(plt.figure(figsize=(5, 3), dpi=100))
        self.original_signal_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        visualization_layout.addWidget(self.original_signal_canvas, 1, 0)
        
        # Gráfico para señal procesada
        processed_label = QLabel("Señal Procesada")
        visualization_layout.addWidget(processed_label, 0, 1)
        
        self.processed_signal_canvas = FigureCanvas(plt.figure(figsize=(5, 3), dpi=100))
        self.processed_signal_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        visualization_layout.addWidget(self.processed_signal_canvas, 1, 1)
        
        # Gráfico para FFT de señal original
        fft_original_label = QLabel("FFT de Señal Original")
        visualization_layout.addWidget(fft_original_label, 2, 0)
        
        self.fft_original_canvas = FigureCanvas(plt.figure(figsize=(5, 3), dpi=100))
        self.fft_original_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        visualization_layout.addWidget(self.fft_original_canvas, 3, 0)
        
        # Gráfico para FFT de señal procesada
        fft_processed_label = QLabel("FFT de Señal Procesada")
        visualization_layout.addWidget(fft_processed_label, 2, 1)
        
        self.fft_processed_canvas = FigureCanvas(plt.figure(figsize=(5, 3), dpi=100))
        self.fft_processed_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        visualization_layout.addWidget(self.fft_processed_canvas, 3, 1)
        
        visualization_group.setLayout(visualization_layout)
        return visualization_group
        
    def on_filter_type_changed(self, filter_type):
        """
        Maneja el cambio en el tipo de filtro seleccionado.
        
        Muestra u oculta el slider de frecuencia alta dependiendo de
        si se selecciona el filtro pasa-banda.
        
        Args:
            filter_type (str): El tipo de filtro seleccionado.
        """
        show_high_freq = filter_type.lower() == 'pasa-banda'
        self.set_high_freq_visibility(show_high_freq)
        
        # Ajustar etiqueta del primer slider
        if show_high_freq:
            self.freq_low_label.setText('Frecuencia de corte baja:')
        else:
            self.freq_low_label.setText('Frecuencia de corte:')
        
    def set_high_freq_visibility(self, visible):
        """
        Establece la visibilidad de los componentes del slider de frecuencia alta.
        
        Args:
            visible (bool): Indica si los componentes deben ser visibles.
        """
        self.freq_high_label.setVisible(visible)
        self.freq_high_slider.setVisible(visible)
        self.freq_high_value.setVisible(visible)
        
    def update_freq_labels(self):
        """
        Actualiza las etiquetas que muestran las frecuencias de corte seleccionadas.
        """
        self.freq_low_value.setText(f'{self.freq_low_slider.value()} Hz')
        self.freq_high_value.setText(f'{self.freq_high_slider.value()} Hz')
        
        # Asegurar que la frecuencia alta sea mayor que la baja
        if self.freq_high_slider.value() <= self.freq_low_slider.value():
            self.freq_high_slider.setValue(self.freq_low_slider.value() + 100)
            
    def load_file(self):
        """
        Abre un diálogo para seleccionar y cargar un archivo de audio.
        
        Después de cargar el archivo, actualiza las variables de audio
        y muestra la señal original y su FFT en los canvas correspondientes.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Cargar Archivo de Audio", 
            "", 
            "Audio Files (*.wav *.mp3)", 
            options=options
        )
        
        if file_name:
            self.audio_data, self.sample_rate = self.load_audio(file_name)
            
            # Resetear límites de ejes
            self.y_limits_signal = None
            self.y_limits_fft = None
            
            # Mostrar señal original y su FFT
            self.plot_signal(self.audio_data, self.original_signal_canvas, "Señal Original")
            
            # Calcular y mostrar FFT de la señal original
            self.fft_original = self.apply_transform_to_audio(self.audio_data)
            self.plot_fft(self.fft_original, self.fft_original_canvas, "FFT de Señal Original")
            
            # Limpiar gráficos de señal procesada
            self.clear_canvas(self.processed_signal_canvas)
            self.clear_canvas(self.fft_processed_canvas)
            self.processed_audio = None
            self.fft_processed = None
    
    def process_signal(self):
        """
        Procesa la señal aplicando el filtro y calculando la transformada.
        
        Este método unifica las funcionalidades de aplicar filtro y
        calcular la transformada en un solo paso.
        """
        if self.audio_data is not None:
            filter_type = self.filter_combo.currentText()
            low_cutoff = self.freq_low_slider.value()
            high_cutoff = self.freq_high_slider.value() if filter_type.lower() == 'pasa-banda' else None
            
            # Aplicar filtro
            self.processed_audio = self.apply_filter_to_audio(
                self.audio_data, 
                filter_type, 
                low_cutoff, 
                high_cutoff,
                self.sample_rate
            )
            
            # Calcular transformadas
            if self.fft_original is None:
                self.fft_original = self.apply_transform_to_audio(self.audio_data)
            
            self.fft_processed = self.apply_transform_to_audio(self.processed_audio)
            
            # Determinar los límites del eje Y para las señales
            if self.y_limits_signal is None:
                max_val_original = np.max(np.abs(self.audio_data))
                max_val_processed = np.max(np.abs(self.processed_audio))
                max_val = max(max_val_original, max_val_processed) * 1.1  # 10% de margen
                self.y_limits_signal = (-max_val, max_val)
            
            # Determinar los límites del eje Y para las FFTs
            if self.y_limits_fft is None:
                max_val_original = np.max(self.fft_original[:len(self.fft_original)//2])
                max_val_processed = np.max(self.fft_processed[:len(self.fft_processed)//2])
                max_val = max(max_val_original, max_val_processed) * 1.1  # 10% de margen
                self.y_limits_fft = (0, max_val)
            
            # Actualizar las gráficas con los límites definidos
            self.plot_signal(self.audio_data, self.original_signal_canvas, "Señal Original", self.y_limits_signal)
            self.plot_signal(self.processed_audio, self.processed_signal_canvas, "Señal Procesada", self.y_limits_signal)
            self.plot_fft(self.fft_original, self.fft_original_canvas, "FFT de Señal Original", self.y_limits_fft)
            self.plot_fft(self.fft_processed, self.fft_processed_canvas, "FFT de Señal Procesada", self.y_limits_fft)
            
    def save_result(self):
        """
        Guarda el audio procesado en un archivo.
        
        Abre un diálogo para seleccionar la ubicación y el formato
        del archivo donde se guardará el audio procesado.
        """
        if self.processed_audio is not None:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, 
                "Guardar Archivo de Audio", 
                "", 
                "Audio Files (*.wav *.mp3 *.aac)", 
                options=options
            )
            
            if file_name:
                self.save_audio(file_name, self.processed_audio, self.sample_rate)
                
    def play_filtered_sound(self):
        """
        Reproduce el audio procesado.
        
        Guarda temporalmente el audio procesado y lo reproduce
        utilizando QSound.
        """
        if self.processed_audio is not None:
            temp_filtered_file = "temp_filtered.wav"
            sf.write(temp_filtered_file, self.processed_audio, self.sample_rate)
            QSound.play(temp_filtered_file)
            
    def plot_signal(self, data, canvas, title, y_limits=None):
        """
        Grafica los datos de señal en el canvas especificado con límites uniformes opcionales.
        
        Args:
            data (numpy.ndarray): Los datos de señal a graficar.
            canvas (FigureCanvas): El canvas donde se mostrará la gráfica.
            title (str): Título del gráfico.
            y_limits (tuple, optional): Límites del eje Y (min, max).
        """
        fig = canvas.figure
        fig.clear()
        
        ax = fig.add_subplot(111)
        ax.plot(data, color='#3498db', linewidth=1)
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Muestras')
        ax.set_ylabel('Amplitud')
        
        # Aplicar límites si están definidos
        if y_limits is not None:
            ax.set_ylim(y_limits)
        
        fig.tight_layout()
        canvas.draw()
        
    def plot_fft(self, fft_data, canvas, title, y_limits=None):
        """
        Grafica los datos de la FFT en el canvas especificado con límites uniformes opcionales.
        
        Args:
            fft_data (numpy.ndarray): Los datos de la FFT a graficar.
            canvas (FigureCanvas): El canvas donde se mostrará la gráfica.
            title (str): Título del gráfico.
            y_limits (tuple, optional): Límites del eje Y (min, max).
        """
        fig = canvas.figure
        fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Mostrar solo la primera mitad del espectro (frecuencias positivas)
        n = len(fft_data)
        half_n = n // 2
        
        # Crear vector de frecuencias si sample_rate está disponible
        if self.sample_rate:
            freq = np.fft.fftfreq(n, d=1/self.sample_rate)[:half_n]
            ax.plot(freq, fft_data[:half_n], color='#e74c3c', linewidth=1)
            ax.set_xlabel('Frecuencia (Hz)')
        else:
            ax.plot(fft_data[:half_n], color='#e74c3c', linewidth=1)
            ax.set_xlabel('Índice de frecuencia')
            
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel('Magnitud')
        
        # Aplicar límites si están definidos
        if y_limits is not None:
            ax.set_ylim(y_limits)
        
        fig.tight_layout()
        canvas.draw()
        
    def clear_canvas(self, canvas):
        """
        Limpia el canvas especificado.
        
        Args:
            canvas (FigureCanvas): El canvas a limpiar.
        """
        fig = canvas.figure
        fig.clear()
        canvas.draw()
        
    def load_audio(self, file_path):
        """
        Carga un archivo de audio desde la ruta especificada.
        
        Args:
            file_path (str): Ruta al archivo de audio a cargar.
            
        Returns:
            tuple: Par de (datos_audio, frecuencia_muestreo).
        """
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
        
    def save_audio(self, file_path, y, sr):
        """
        Guarda los datos de audio en un archivo.
        
        Args:
            file_path (str): Ruta donde guardar el archivo.
            y (numpy.ndarray): Datos de audio a guardar.
            sr (int): Frecuencia de muestreo del audio.
        """
        sf.write(file_path, y, sr)
        
    def apply_filter_to_audio(self, y, filter_type, low_cutoff, high_cutoff, sr):
        """
        Aplica un filtro a los datos de audio.
        
        Args:
            y (numpy.ndarray): Datos de audio a filtrar.
            filter_type (str): Tipo de filtro a aplicar ('pasa-bajas', 'pasa-altas', 'pasa-banda').
            low_cutoff (int): Frecuencia de corte inferior.
            high_cutoff (int, optional): Frecuencia de corte superior para filtro pasa-banda.
            sr (int): Frecuencia de muestreo del audio.
            
        Returns:
            numpy.ndarray: Los datos de audio filtrados.
        """
        nyquist = 0.5 * sr
        
        # Convertir los tipos de filtro a lowercase para comparación insensible a mayúsculas
        filter_type_lower = filter_type.lower()
        
        if filter_type_lower == 'pasa-bajas':
            normal_cutoff = low_cutoff / nyquist
            b, a = butter(4, normal_cutoff, btype='low', analog=False)
        elif filter_type_lower == 'pasa-altas':
            normal_cutoff = low_cutoff / nyquist
            b, a = butter(4, normal_cutoff, btype='high', analog=False)
        else:  # Pasa-banda
            low_normal_cutoff = low_cutoff / nyquist
            high_normal_cutoff = high_cutoff / nyquist
            # Asegurar que high > low
            if high_normal_cutoff <= low_normal_cutoff:
                high_normal_cutoff = low_normal_cutoff + 0.1
            # Asegurar que high < 1 (límite de Nyquist)
            high_normal_cutoff = min(high_normal_cutoff, 0.99)
            b, a = butter(4, [low_normal_cutoff, high_normal_cutoff], btype='band', analog=False)
            
        y_filtered = lfilter(b, a, y)
        return y_filtered
        
    def apply_transform_to_audio(self, y):
        """
        Aplica la transformada de Fourier a los datos de audio.
        
        Args:
            y (numpy.ndarray): Datos de audio a transformar.
            
        Returns:
            numpy.ndarray: Magnitud de la transformada de Fourier.
        """
        return np.abs(np.fft.fft(y))
        
    def apply_stylesheet(self):
        """
        Aplica estilos CSS a la aplicación para mejorar su apariencia.
        """
        style = """
        QMainWindow {
            background-color: #f5f5f5;
        }

        QGroupBox {
            background-color: white;
            border-radius: 8px;
            border: 1px solid #dcdcdc;
            margin-top: 12px;
            font-weight: bold;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
        }

        QPushButton {
            background-color: #3498db;
            color: white;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            border: none;
        }

        QPushButton:hover {
            background-color: #2980b9;
        }

        QPushButton:pressed {
            background-color: #1c6ea4;
        }

        QComboBox {
            border: 1px solid #dcdcdc;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
            selection-background-color: #3498db;
            min-width: 150px;
        }

        QComboBox::drop-down {
            border: none;
            width: 20px;
        }

        QSlider::groove:horizontal {
            border: 1px solid #dcdcdc;
            height: 8px;
            background: white;
            margin: 2px 0;
            border-radius: 4px;
        }

        QSlider::handle:horizontal {
            background: #3498db;
            border: none;
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }

        QSlider::handle:horizontal:hover {
            background: #2980b9;
        }

        QLabel {
            color: #333333;
        }
        """
        self.setStyleSheet(style)


def main():
    """
    Función principal que inicializa y ejecuta la aplicación.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()