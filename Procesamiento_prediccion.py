import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from zipfile import ZipFile
from sklearn.metrics import r2_score, mean_squared_error

# Descargar y extraer el archivo CSV
url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
response = requests.get(url)
with ZipFile(BytesIO(response.content)) as thezip:
    with thezip.open('jena_climate_2009_2016.csv') as thefile:
        df = pd.read_csv(thefile)

# Formato tiempo
df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

# Se eliminan las columnas que no se usarán
df = df.drop(columns=['Tpot (K)', 'sh (g/kg)', 'Tdew (degC)', 'VPmax (mbar)', 'VPact (mbar)', 
                     'VPdef (mbar)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'max. wv (m/s)'])

wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

for column in df.columns:
    qs = df[column].quantile([0.25, 0.5, 0.75]).values
    q1 = qs[0]
    q2 = qs[1]
    q3 = qs[2]
    iqr = q3 - q1
    min_val = q1 - 1.5 * iqr
    max_val = q3 + 1.5 * iqr

    df[column] = np.where(df[column] > max_val, max_val, df[column])
    df[column] = np.where(df[column] < min_val, min_val, df[column])

# Convertir la columna de dirección y velocidad del viento en un vector de viento
wv_ = df.pop('wv (m/s)')
wd_rad = df.pop('wd (deg)') * np.pi / 180

# Calcular las componentes x e y del viento
df['Wx'] = wv_ * np.cos(wd_rad)
df['Wy'] = wv_ * np.sin(wd_rad)

# Convertir la columna Date Time en segundos
df['Seconds'] = df.index.map(pd.Timestamp.timestamp)

# Utilizar sin y cos para convertir el tiempo en señales claras de "Hora del día" y "Hora del año"
day = 60 * 60 * 24
year = 365.2425 * day

df['Day sin'] = np.sin(df['Seconds'] * (2 * np.pi / day))
df['Day cos'] = np.cos(df['Seconds'] * (2 * np.pi / day))
df['Year sin'] = np.sin(df['Seconds'] * (2 * np.pi / year))
df['Year cos'] = np.cos(df['Seconds'] * (2 * np.pi / year))
df = df.drop('Seconds', axis=1)

df1 = df.iloc[:400000]  # Primer DataFrame con las primeras 400,000 filas

def df_to_X_y3(df, window_size=10):  # depende del número de columnas del dataset (columnas+1)
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)
        label = [df_as_np[i + window_size][0], df_as_np[i + window_size][1], 
                 df_as_np[i + window_size][2], df_as_np[i + window_size][3], 
                 df_as_np[i + window_size][4]]
        y.append(label)
    return np.array(X), np.array(y)

X3, y3 = df_to_X_y3(df1)

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
X3_train, y3_train = X3[0:int(n * 0.8)], y3[0:int(n * 0.8)]  # 80%
X3_val, y3_val = X3[int(n * 0.8):int(n * 0.9)], y3[int(n * 0.8):int(n * 0.9)]  # 10%
X3_test, y3_test = X3[int(n * 0.9):], y3[int(n * 0.9):]  # 10%

# Calcular las medias y desviaciones estándar para cada característica
means3 = np.mean(X3_train, axis=(0, 1))
stds3 = np.std(X3_train, axis=(0, 1))

# Función para preprocesar el conjunto X
def preprocess3(X):
    for i in range(X.shape[2]):
        X[:, :, i] = (X[:, :, i] - means3[i]) / stds3[i]

# Función para preprocesar el conjunto y (output)
def preprocess_output3(y):
    for i in range(y.shape[1]):
        y[:, i] = (y[:, i] - means3[i]) / stds3[i]
    return y

preprocess3(X3_train)
preprocess3(X3_val)
preprocess3(X3_test)

preprocess_output3(y3_train)
preprocess_output3(y3_val)
preprocess_output3(y3_test)

# Convertir a tensores de PyTorch
X3_train_t = torch.tensor(X3_train, dtype=torch.float32)
y3_train_t = torch.tensor(y3_train, dtype=torch.float32)
X3_val_t = torch.tensor(X3_val, dtype=torch.float32)
y3_val_t = torch.tensor(y3_val, dtype=torch.float32)
X3_test_t = torch.tensor(X3_test, dtype=torch.float32)
y3_test_t = torch.tensor(y3_test, dtype=torch.float32)

# Crear un DataLoader para entrenamiento y validación
train_dataset = TensorDataset(X3_train_t, y3_train_t)
val_dataset = TensorDataset(X3_val_t, y3_val_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Definir el modelo LSTM en PyTorch
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=9, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 5)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Seleccionamos la última salida de la secuencia
        x = torch.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        return x

# Instanciar el modelo, la función de pérdida y el optimizador
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Entrenamiento del modelo
epochs = 10
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Validación
model.eval()
val_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader)}")

# Prueba del modelo
model.eval()
with torch.no_grad():
    pred = model(X3_test_t).numpy()

model_path_full = r"C:\Users\estef\OneDrive\Documentos\UPTC\Semestre 11\PROYECTO GRADO\Codigo prediccion PYTHON\model_complete.pth"
torch.save(model, model_path_full)

print('R-Squared:', r2_score(y3_test, pred))
print('Mean squared error:', mean_squared_error(pred, y3_test))

# Cargar el dataset
file_path = r"C:\Users\estef\OneDrive\Documentos\UPTC\Semestre 11\PROYECTO GRADO\DATOS\datos_jena_divididos\dataset_combined.csv"
df_n = pd.read_csv(file_path)

# Eliminar las columnas que no se usarán
df_n = df_n.drop(columns=['Day sin', 'Day cos', 'Year sin', 'Year cos'])

# Rango de fechas
start_date = '2016-08-08 04:50:00'
end_date = '2017-01-01 00:00:00'

# Generar un rango de fechas con el formato solicitado
date_range = pd.date_range(start=start_date, end=end_date, freq='10T')
df_n['timestamp'] = date_range

# Formatear los datos temporales
df_n.index = pd.to_datetime(df_n['timestamp'], format='%Y-%m-%d %H:%M:%S')
date_time = pd.to_datetime(df_n.pop('timestamp'), format='%Y-%m-%d %H:%M:%S')

# Crear características sin/cos para "hora del día" y "hora del año"
df_n['Seconds'] = df_n.index.map(pd.Timestamp.timestamp)
day = 60 * 60 * 24
year = 365.2425 * day

df_n['Day sin'] = np.sin(df_n['Seconds'] * (2 * np.pi / day))
df_n['Day cos'] = np.cos(df_n['Seconds'] * (2 * np.pi / day))
df_n['Year sin'] = np.sin(df_n['Seconds'] * (2 * np.pi / year))
df_n['Year cos'] = np.cos(df_n['Seconds'] * (2 * np.pi / year))
df_n = df_n.drop('Seconds', axis=1)

# Función para generar X y y
def df_to_X_y(df, window_size=10):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)
        label = [df_as_np[i + window_size][0], df_as_np[i + window_size][1],
                 df_as_np[i + window_size][2], df_as_np[i + window_size][3],
                 df_as_np[i + window_size][4]]
        y.append(label)
    return np.array(X), np.array(y)

X_pred_seq, y_pred_seq = df_to_X_y(df_n)

# Calcular medias y desviaciones estándar
means = np.mean(X_pred_seq, axis=(0, 1))
stds = np.std(X_pred_seq, axis=(0, 1))

# Función para normalizar las entradas
def preprocess_p(X):
    for i in range(X.shape[2]):
        X[:, :, i] = (X[:, :, i] - means[i]) / stds[i]

# Función para normalizar las salidas
def preprocess_output_p(y):
    for i in range(y.shape[1]):
        y[:, i] = (y[:, i] - means[i]) / stds[i]
    return y

preprocess_p(X_pred_seq)
preprocess_output_p(y_pred_seq)

# Convertir los datos a tensores
X_pred_t = torch.tensor(X_pred_seq, dtype=torch.float32)
y_pred_t = torch.tensor(y_pred_seq, dtype=torch.float32)

# Definir un dataset y DataLoader para manejar los datos
dataset = TensorDataset(X_pred_t, y_pred_t)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Evaluar el modelo
model.eval()
with torch.no_grad():
    pred_f = model(X_pred_t).numpy()

# Calcular R-squared y MSE
print('R-Squared:', r2_score(y_pred_seq, pred_f))
print('Mean squared error:', mean_squared_error(pred_f, y_pred_seq))

# Postprocesar las predicciones
def postprocess(arr, idx):
    return (arr * stds[idx]) + means[idx]

# Asignar las columnas a las predicciones
p_arr = pred_f[:, 0]  # Columna para presión
t_arr = pred_f[:, 1]  # Columna para temperatura
sh_arr = pred_f[:, 2]  # Columna para humedad específica
wx_arr = pred_f[:, 3]  # Columna para componente Wx
wy_arr = pred_f[:, 4]  # Columna para componente Wy

# Desnormalizar las predicciones
p_preds_n = postprocess(p_arr, 0)
t_preds_n = postprocess(t_arr, 1)
sh_preds_n = postprocess(sh_arr, 2)
wx_preds_n = postprocess(wx_arr, 3)
wy_preds_n = postprocess(wy_arr, 4)

# Funciones para calcular velocidad y dirección del viento
def calculate_wind_speed(Wx, Wy):
    return np.sqrt(Wx ** 2 + Wy ** 2)

def calculate_wind_direction(Wx, Wy):
    wd_rad = np.arctan2(Wy, Wx)
    wd_deg = np.degrees(wd_rad)
    wd_deg = (wd_deg + 360) % 360
    return wd_deg

window_size = 10

# Calcular velocidad y dirección del viento
wv_preds_n = calculate_wind_speed(wx_preds_n, wy_preds_n)
wd_preds_n = calculate_wind_direction(wx_preds_n, wy_preds_n)

# Crear DataFrame con los resultados
results_df = pd.DataFrame({
    'Fecha': date_range[window_size:],
    'Presion': p_preds_n,
    'Temperatura': t_preds_n,
    'Humedad': sh_preds_n,
    'Velocidad Viento': wv_preds_n,
    'Direccion Viento': wd_preds_n,
})

# Establecer 'Fecha' como índice y redondear los valores
results_df.set_index('Fecha', inplace=True)
results_df = results_df.round(2)

# Guardar el DataFrame
output_path = r"C:\Users\estef\OneDrive\Documentos\UPTC\Semestre 11\PROYECTO GRADO\DATOS\PREDICCIONES\predicciones_torch.csv"
results_df.to_csv(output_path, index=False)


print(results_df)
