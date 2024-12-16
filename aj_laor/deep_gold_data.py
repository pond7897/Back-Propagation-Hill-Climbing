import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. โหลดข้อมูลจากไฟล์ CSV
data = pd.read_csv('./archive/XAU_1d_data_2004_to_2024-09-20.csv')  # เปลี่ยนชื่อไฟล์ตามที่คุณมี

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

prices = data['Close'].values.reshape(-1, 1)

# 2. การเตรียมข้อมูล
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

def create_dataset(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 30
X, y = create_dataset(prices_scaled, look_back)
X = X.reshape(X.shape[0], X.shape[1], 1)  # เปลี่ยนรูปเป็น 3D สำหรับ LSTM

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. ฟังก์ชันสุ่มน้ำหนักและ feed forward

def initialize_random_weights():
    return np.random.uniform(-0.5, 0.5, size=(look_back, 50))

def calculate_error(model, X, y):
    predictions = model.predict(X)
    return np.mean(np.square(y - predictions.flatten()))

# 4. ฟังก์ชันสร้างโมเดล

def create_model(initializer):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(look_back, 1), kernel_initializer=initializer),
        tf.keras.layers.LSTM(50, kernel_initializer=initializer),
        tf.keras.layers.Dense(1, kernel_initializer=initializer)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# 5. แบ่ง Cluster ของ Error
initial_weights = [initialize_random_weights() for _ in range(5)]  # Random weights 5 ชุด
errors = []

for weights in initial_weights:
    model = create_model(tf.keras.initializers.RandomNormal(mean=np.mean(weights), stddev=np.std(weights)))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # Train เบื้องต้น
    error = calculate_error(model, X_train, y_train)
    errors.append(error)

# ใช้ KMeans แบ่ง Cluster
errors = np.array(errors).reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(errors)

# เลือก Cluster ที่ Error น้อยที่สุด
best_cluster_idx = np.argmin(kmeans.cluster_centers_)
selected_weights = [initial_weights[i] for i in range(len(clusters)) if clusters[i] == best_cluster_idx]

# คำนวณค่าเฉลี่ยน้ำหนัก
average_weights = np.mean(selected_weights, axis=0)

# 6. Train ต่อด้วยน้ำหนักเฉลี่ย
final_model = create_model(tf.keras.initializers.RandomNormal(mean=np.mean(average_weights), stddev=np.std(average_weights)))
final_history = final_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 7. เปรียบเทียบ Xavier และ He Initializers
initializers = {
    'He_normal': tf.keras.initializers.HeNormal(),
    'Xavier': tf.keras.initializers.GlorotUniform()
}

histories = {}

for name, initializer in initializers.items():
    model = create_model(initializer)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    histories[name] = history

# แสดงกราฟเปรียบเทียบ Loss
plt.figure(figsize=(12, 6))
plt.plot(final_history.history['val_loss'], label='Cluster-based Initializer')
for name, history in histories.items():
    plt.plot(history.history['val_loss'], label=f'{name} Initializer')
plt.title('Comparison of Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
