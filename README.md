Transmision_datos.ino  (ARDUINO)

Este algoritmo es responsable de recoger los datos de los sensores encargados de medir temperatura, humedad, presión atmosférica, velocidad y dirección del viento, precipitación y radiación solar en
intervalos de 6 segundos. El intervalo se configura según la frecuencia de medición requerida para cada variable. Para transmitir los datos, se utiliza un arreglo de tamaño constante tipo byte, donde se
empaquetan todos los datos de los sensores. Este arreglo tiene una longitud de 40 bytes, ya que cada dato medido es de tipo float (32 bits o 4 bytes por cada valor). Cada lectura de sensor se convierte a
su formato binario y se almacena secuencialmente en posiciones específicas dentro de dicho arreglo, de manera que al momento de ser recibidos por otro dispositivo no haya pérdida de información.
Los datos se transmiten en tiempo real al dispositivo de tratamiento donde estos se procesarán. 

Adquisicion_datos.py  (RASPBERRY)

Este codigo captura los datos generados por los sensores de la estacion meteorologica en intervalos regulares de 6 segundos, los almacena en un archivo .CSV para su posterior análisis y también los 
guarda en una base de datos SQL para su visualización. 

Una vez recibidos los datos estos son desempaquetados para convertir los bytes a valores flotantes y que posteriormente puedan ser procesados y utilizados, el código los almacena de dos formas. 
Primero, los datos se organizan en un `DataFrame` de pandas y se guardan en un archivo CSV llamado `datos_climaticos.csv`. Si el archivo no existe, el sistema lo crea, y si ya existe, los datos nuevos 
se añaden. Además, se implementa un mecanismo para guardar los datos en este archivo cada 5 minutos, asegurando que la información se almacene periódicamente y esté disponible para futuros análisis y 
modelado de predicciones. Por otro lado, el código también almacena los datos en una base de datos MySQL para facilitar su visualización y análisis. Se accede a la base de datos existente en la Raspberry®,
insertar los datos en la tabla correspondiente y luego cerrar la conexión. Este almacenamiento en la base de datos es crucial para acceder a los datos de manera más dinámica, por ejemplo, en una plataforma
web o servidor de visualización.

Procesamiento_prediccion.py  (RASPBERRY)

Este codigo inicialmente se encarga de preprocesar y procesar el conjunto de datos Jena Climate Dataset, es un paso fundamental para garantizar la eficacia del modelo de predicción climática. 

Una vez transformados los datos, se dividen en conjuntos de entrenamiento 80%, validación 10% y prueba 10% para estructurar adecuadamente el proceso de modelado. Esta división es crucial para evaluar 
el rendimiento del modelo y evitar el sobreajuste, asegurando que el modelo generalice bien a datos no vistos.
Para el entrenamiento del modelo de predicción climática basado en inteligencia artificial, se seleccionó el 95% del conjunto de datos Jena Climate Dataset, equivalente a 400.000 filas de datos entre 
01 de enero de 2009 y 07 de agosto de 2016. 

Posteriormente se establece un modelo para la predicción de variables climáticas, cuyo objetivo es prever con un error no mayor al 10% cada una de las siguientes variables: presión atmosférica en 
hectopascales (hPa), temperatura en grados Celsius (°C), porcentaje de humedad relativa (%RH), velocidad del viento en metros por segundo (m/s) y dirección del viento en grados (°).
Para generar predicciones meteorológicas se requiere una gran cantidad de datos históricos para capturar correctamente las variaciones y tendencias climáticas, por esta razón se optó por utilizar 
en 5% restante del conjunto de datos Jena Climate Dataset (dataset_combined.csv), equivalente a 20.995 filas de datos entre 08 de agosto de 2016 y 31 de diciembre de 2016.
