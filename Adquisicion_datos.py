from time import sleep, time
from SX127x.LoRa import *
from SX127x.board_config import BOARD
import struct
import sys
import pandas as pd
from datetime import datetime
import os
import MySQLdb
from datetime import datetime

BOARD.setup()

class LoRaRcvCont(LoRa):
    def __init__(self, verbose=False):
        super(LoRaRcvCont, self).__init__(verbose)
        self.set_mode(MODE.SLEEP)
        self.set_dio_mapping([0] * 6)

        # Inicializar el DataFrame vacío con las columnas deseadas
        self.filename = 'datos_climaticos.csv'
        self.data_df = pd.DataFrame(columns=['Fecha y Hora', 'Temperatura', 'Humedad', 'Presion', 'Altitud', 
                                             'Lluvia Acumulada', 'Tasa de Lluvia', 'Radiacion', 
                                             'Velocidad del Viento', 'Direccion del Viento'])

        # Cargar el DataFrame desde el archivo CSV si existe
        if os.path.isfile(self.filename):
            self.data_df = pd.read_csv(self.filename)
        else:
            self.data_df.to_csv(self.filename, index=False)  # Crear archivo CSV vacío si no existe

        # Variables de tiempo para guardar
        self.last_save_time = time()   # Última vez que se guardaron datos
        self.save_interval = 300       # Intervalo para guardar en segundos (5 minutos)
        self.received_data = {}        # Diccionario temporal para almacenar el último paquete de datos

    def start(self):
        self.reset_ptr_rx()
        self.set_mode(MODE.RXCONT)
        while True:
            current_time = time()

            # Guardar los datos en CSV cada 5 minutos
            if current_time - self.last_save_time >= self.save_interval:
                self.save_data_to_csv()
                self.last_save_time = current_time  # Actualizar tiempo del último guardado

            sleep(0.5)  # Pequeña pausa para no sobrecargar la CPU

    def on_rx_done(self):
        print("\nReceived: ")
        self.clear_irq_flags(RxDone=1)

        # Leer el payload desde el dispositivo
        payload = self.read_payload(nocheck=True)

        # Asegurarse de que el payload esté en formato de bytes
        if isinstance(payload, list):
            payload = bytes(payload)

        print(f"Tamaño payload: {len(payload)} bytes")

        # Verificar si el tamaño del payload es el esperado (40 bytes para 10 floats)
        if len(payload) >= 40:  # 10 floats * 4 bytes cada uno = 40 bytes
            try:
                data = payload[-40:]  # Los últimos 40 bytes contienen los datos
                # Desempaquetar los datos binarios a floats
                a, temp, hum, pres, alt, llu, tas, irr, vel, dire = struct.unpack('ffffffffff', data)

                # Redondear cada variable a dos decimales
                temp = round(temp, 2)
                hum = round(hum, 2)
                pres = round(pres, 2)
                alt = round(alt, 2)
                llu = round(llu, 2)
                tas = round(tas, 2)
                irr = round(irr, 2)
                vel = round(vel, 2)
                dire = round(dire, 2)

                # Obtener la fecha y hora actual
                fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Crear un diccionario con los datos recibidos
                self.received_data = {
                    'Fecha y Hora': fecha_hora,
                    'Temperatura': temp,
                    'Humedad': hum,
                    'Presión': pres,
                    'Altitud': alt,
                    'Lluvia Acumulada': llu,
                    'Tasa de Lluvia': tas,
                    'Radiación': irr,
                    'Velocidad del Viento': vel,
                    'Dirección del Viento': dire
                }

                # Mostrar los datos inmediatamente
                self.display_data()
                # Guardar los datos en la base de datos
                self.save_data_to_db()

            except struct.error as e:
                print(f"Error al desempaquetar los datos: {e}")
        else:
            print("Error: tamaño inesperado del payload")

        # Configura el dispositivo de radio de nuevo en modo de reposo y luego modo de recepción continua
        self.set_mode(MODE.SLEEP)
        self.reset_ptr_rx()
        self.set_mode(MODE.RXCONT)

    def display_data(self):
        """Función para mostrar los datos inmediatamente después de recibirlos."""
        if self.received_data:
            # Mostrar el último registro recibido en pantalla
            last_data = self.received_data
            print("\nDatos recibidos:")
            print(f"Fecha y Hora: {last_data['Fecha y Hora']}")
            print(f"Temperatura: {last_data['Temperatura']:.2f} °C")
            print(f"Humedad: {last_data['Humedad']:.2f} %")
            print(f"Presión atmosférica: {last_data['Presión']:.2f} hPa")
            print(f"Altitud: {last_data['Altitud']:.2f} m")
            print(f"Lluvia acumulada: {last_data['Lluvia Acumulada']:.2f} mm")
            print(f"Tasa de lluvia: {last_data['Tasa de Lluvia']:.2f} mm/h")
            print(f"Radiación: {last_data['Radiación']:.2f} W/m2")
            print(f"Velocidad del viento: {last_data['Velocidad del Viento']:.2f} m/s")
            print(f"Dirección del viento: {last_data['Dirección del Viento']:.2f} °")

    def save_data_to_csv(self):
        """Función para guardar el último paquete de datos en el archivo CSV cada 5 minutos."""
        if self.received_data:
            # Convertir el último paquete de datos a DataFrame
            new_data_df = pd.DataFrame([self.received_data])

            # Añadir los nuevos datos al DataFrame existente
            self.data_df = pd.concat([self.data_df, new_data_df], ignore_index=True)

            # Guardar el DataFrame en un archivo CSV
            self.data_df.to_csv(self.filename, index=False)
            print(f"\nDatos guardados en '{self.filename}' a las {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Vaciar el diccionario de datos recibidos después de guardarlos
            self.received_data = {}

    def save_data_to_db(self):
        """Función para guardar los datos en una base de datos MySQL."""
        if self.received_data:
            try:
                # Conectar a la base de datos
                db = MySQLdb.connect(host="localhost", user="i2e", passwd="1234", db="datos_clima")
                cur = db.cursor()

                # Convertir la fecha y hora a formato Unix timestamp (segundos desde epoch)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Imprimir los valores para depuración
                print("Valores a insertar en la base de datos:")
                for key, value in self.received_data.items():
                    print(f"{key}: {value}")

                # Insertar datos en la nueva tabla
                cur.execute('''INSERT INTO Datos (time, temp, hum, pres, alt, llu, tas, vel, dire, irr)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                ''', (
                    timestamp,
                    self.received_data.get('Temperatura', None),
                    self.received_data.get('Humedad', None),
                    self.received_data.get('Presion', None),
                    self.received_data.get('Altitud', None),
                    self.received_data.get('Lluvia Acumulada', None),
                    self.received_data.get('Tasa de Lluvia', None),
                    self.received_data.get('Velocidad del Viento', None),
                    self.received_data.get('Direccion del Viento', None),
                    self.received_data.get('Radiacion', None)
                ))

                # Confirmar cambios
                db.commit()
                print(f"Datos guardados en la base de datos")

            except MySQLdb.Error as e:
                print(f"Error al guardar en la base de datos: {e}")
                db.rollback()  # Deshacer cambios en caso de error

            finally:
                # Cerrar la conexión
                cur.close()
                db.close()

lora = LoRaRcvCont(verbose=False)
lora.set_mode(MODE.STDBY)
lora.set_pa_config(pa_select=1)

try:
    lora.start()
except KeyboardInterrupt:
    sys.stdout.flush()
    print("")
    sys.stderr.write("KeyboardInterrupt\n")
finally:
    sys.stdout.flush()
    print("")
    lora.set_mode(MODE.SLEEP)
    BOARD.teardown()

