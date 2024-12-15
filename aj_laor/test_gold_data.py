import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. โหลดข้อมูลจากไฟล์ CSV
data = pd.read_csv('./archive/XAU_1d_data_2004_to_2024-09-20.csv')  # เปลี่ยนชื่อไฟล์ตามที่คุณมี

# สมมติว่าข้อมูลมีคอลัมน์ 'Date' และ 'Price'
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# ใช้เฉพาะคอลัมน์ราคาทองคำ
prices = data['Close'].values.reshape(-1, 1)

# 2. การเตรียมข้อมูล
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# สร้างชุดข้อมูลสำหรับการพยากรณ์ 30 วัน
def create_dataset(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 30
X, y = create_dataset(prices_scaled, look_back)
X = X.reshape(X.shape[0], X.shape[1], 1)  # เปลี่ยนรูปเป็น 3D สำหรับ LSTM

# แบ่งข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. สร้างฟังก์ชันโมเดล

def create_model(initializer):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(look_back, 1), kernel_initializer=initializer),
        tf.keras.layers.LSTM(50, kernel_initializer=initializer),
        tf.keras.layers.Dense(1, kernel_initializer=initializer)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# 4. เปรียบเทียบ He และ Xavier
initializers = {
    'He_normal': tf.keras.initializers.HeNormal(),
    'He_uniform': tf.keras.initializers.HeUniform(),
    'Xavier': tf.keras.initializers.GlorotUniform(),
    'Zeros': tf.keras.initializers.Zeros()
}

histories = {}
models = {}

for name, initializer in initializers.items():
    print(f"Training model with {name} initializer...")
    model = create_model(initializer)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    histories[name] = history
    models[name] = model

# 5. การเปรียบเทียบผลลัพธ์
plt.figure(figsize=(12, 6))
for name, history in histories.items():
    plt.plot(history.history['val_loss'], label=f'{name} Validation Loss')
plt.title('Comparison of Validation Loss Initialization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 6. การพยากรณ์ราคาทองคำ 30 วันถัดไป (ตัวอย่าง)
predictions = {}
for name, model in models.items():
    predicted = model.predict(X_test)
    predictions[name] = scaler.inverse_transform(predicted.reshape(-1, 1))

# ดูตัวอย่างผลลัพธ์
for name, predicted in predictions.items():
    print(f"{name} Initializer - Predicted Prices: {predicted[:5].flatten()}")