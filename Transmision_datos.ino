#include <SPI.h>
#include <RH_RF95.h>
#include <Arduino.h>
#include <math.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

#define SEALEVELPRESSURE_HPA (1013.25)
Adafruit_BME280 bme; // I2C

// Variables para el sensor de lluvia
const byte pinInterrupcion = 4;
const int intervalo = 1000;  // Intervalo de debounce (ms)
volatile unsigned long tiempoCaida = millis();
volatile double lluviaAcumulada = 0.0;  // Acumulación de lluvia en mm
const double mmPorPulso = 0.2;  // Cada pulso equivale a 0.2 mm de lluvia
volatile double tasaLluvia = 0.0;  // Tasa de lluvia en mm/h

// Variables para reiniciar el conteo de lluvia
unsigned long tiempoInicioLluvia = millis();
const unsigned long intervaloReinicioLluvia = 300000;  // 5 minutos en milisegundos (300,000 ms)
const unsigned long intervaloMuestreoLluvia = 2000;  // 2 segundos en milisegundos (2000 ms)
unsigned long ultimoTiempoMuestreoLluvia = millis();  // Variable para el último muestreo


//Variables radiacion solar
const int sensorPin = A0;  // Pin analógico donde está conectado el sensor
const float voltajePorWpm2 = 0.00167;  // Voltios por W/m² (1.67 mV/W/m²)
const float VREF = 3.0;  // Voltaje de referencia del sensor (0 a 3V)

// Variables de velocidad y dirección del viento
#define PIN_VELOCIDAD_VIENTO 3
#define PIN_DIRECCION_VIENTO A3
#define T 5  // Tiempo de muestreo para el anemómetro
volatile unsigned int rotaciones = 0;
volatile float direccion_instantanea = 0;
unsigned int ultimo_millis_publicacion = 0;

RH_RF95 rf95;

void setup()
{
  //Serial.begin(9600);
  if (!rf95.init()) {
    Serial.println("LoRa init failed");
  }

  // Inicializar BME280
  unsigned status;
  status = bme.begin();
  if (!status) {
    Serial.println("No se puede encontrar el sensor BME280");
  }

  // Configuración del sensor de lluvia
  pinMode(pinInterrupcion, INPUT_PULLUP);
  PCICR |= (1 << PCIE2);      // Habilitar interrupciones en el grupo PCINT2 (pines D0 a D7)
  PCMSK2 |= (1 << PCINT20);   // Habilitar la interrupción en el pin D4 (PCINT20)
  sei();  // Habilitar interrupciones globales

  // Configuración del sensor de velocidad del viento
  pinMode(PIN_VELOCIDAD_VIENTO, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_VELOCIDAD_VIENTO), interrupcionRotacion, FALLING);

}

void loop() {

    // Leer los datos de los sensores BME280
    float temp = bme.readTemperature();
    float hum = bme.readHumidity();
    float pres = bme.readPressure() / 100.0F;
    float alt = bme.readAltitude(SEALEVELPRESSURE_HPA);
    float voltaje = analogRead(sensorPin) * (VREF / 1023.0);  // Conversión de ADC a voltaje real
    float irradiancia = voltaje / voltajePorWpm2;

    // Control del bucle a 1 Hz para el anemómetro (velocidad del viento)
    unsigned long tiempoActual = millis();
    if ((tiempoActual - ultimo_millis_publicacion) > 1000) {
      // Calcular la velocidad del viento (m/s) y resetear el contador
      float velocidad = (rotaciones * 2.25 / T) * 0.44704; // Conversión a m/s
      rotaciones = 0;  // Resetear rotaciones para el siguiente cálculo

      // Actualizar la dirección del viento (en grados)
      direccion_instantanea = map(analogRead(PIN_DIRECCION_VIENTO), 0, 1023, 0, 359);

      // Empaquetar los datos en un arreglo de bytes
      uint8_t message[sizeof(float) * 10];  // Ahora 10 floats
      float valor_vacio = 0.0f;

      memcpy(&message[0], &valor_vacio, sizeof(float));  // Copiar un "valor vacío"
      memcpy(&message[sizeof(float)], &temp, sizeof(float));
      memcpy(&message[sizeof(float) * 2], &hum, sizeof(float));
      memcpy(&message[sizeof(float) * 3], &pres, sizeof(float));
      memcpy(&message[sizeof(float) * 4], &alt, sizeof(float));
      memcpy(&message[sizeof(float) * 5], &lluviaAcumulada, sizeof(float));  
      memcpy(&message[sizeof(float) * 6], &tasaLluvia, sizeof(float)); 
      memcpy(&message[sizeof(float) * 7], &irradiancia, sizeof(float));
      memcpy(&message[sizeof(float) * 8], &velocidad, sizeof(float));  // Velocidad del viento
      memcpy(&message[sizeof(float) * 9], &direccion_instantanea, sizeof(float));  // Dirección del viento

      // Enviar los datos binarios a través de LoRa
      rf95.send(message, sizeof(message));
      rf95.waitPacketSent();

      // Resetear el contador de tiempo
      ultimo_millis_publicacion = tiempoActual;
  }
      // Verificar si han pasado 5 minutos para reiniciar lluviaAcumulada
  if ((tiempoActual - tiempoInicioLluvia) > intervaloReinicioLluvia) {
    lluviaAcumulada = 0.0;  // Reiniciar la acumulación de lluvia
    tasaLluvia = 0.0;  // Reiniciar la tasa de lluvia
    tiempoInicioLluvia = tiempoActual;  // Reiniciar el contador de tiempo
  }

  delay(2000);  // Retraso para la próxima transmisión
}

// ISR para manejar el cambio de estado en el pin D4 (sensor de lluvia)
ISR(PCINT2_vect) {
  unsigned long tiempoActual = millis();

  // Debounce para evitar falsas interrupciones debido a rebotes
  if ((tiempoActual - tiempoCaida) < intervalo) {
    return;
  }

  unsigned long tiempoCaidaAnterior = tiempoActual - tiempoCaida;
  tiempoCaida = tiempoActual;

  // Calcular la tasa de lluvia en mm/h
  tasaLluvia = (mmPorPulso / (tiempoCaidaAnterior / 1000.0)) * 3600.0;  // mm/h

  // Acumular la cantidad de lluvia
  lluviaAcumulada += mmPorPulso;
}

// ISR para el anemómetro (velocidad del viento)
void interrupcionRotacion() {
  rotaciones++;
}
