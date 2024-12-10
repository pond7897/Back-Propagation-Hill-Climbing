import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score

np.random.seed(42)

def clustered_initializer(shape, actual_values, he_init=True, n_clusters=8, mode='best_cluster'):
    if he_init:
        initial_weights = np.random.randn(*shape) * np.sqrt(2 / shape[1])  # He initialization
    else:
        initial_weights = np.random.randn(*shape) * np.sqrt(1 / shape[1])  # Xavier initialization


    actual_values_reshaped = actual_values[:shape[0]].mean(axis=1, keepdims=True) if shape[0] <= actual_values.shape[0] else actual_values.mean(axis=0, keepdims=True)
    
    errors = np.abs(actual_values_reshaped - initial_weights.mean(axis=1, keepdims=True))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clustered_weights = kmeans.fit_predict(errors)


    adjusted_weights = initial_weights.copy()
    if mode == 'best_cluster':

        best_cluster_id = np.argmin([np.mean(errors[clustered_weights == i]) for i in range(n_clusters)])
        best_indices = np.where(clustered_weights == best_cluster_id)[0]
        
        if len(best_indices) < shape[0]:
            cluster_mean = initial_weights[best_indices].mean(axis=0)
            adjusted_weights = np.tile(cluster_mean, (shape[0], 1)) 
        else:
            adjusted_weights = initial_weights[best_indices]

    elif mode == 'average':
        best_cluster_id = np.argmin([np.mean(errors[clustered_weights == i]) for i in range(n_clusters)])
        best_indices = np.where(clustered_weights == best_cluster_id)[0]
        cluster_mean = initial_weights[best_indices].mean(axis=0)
        adjusted_weights = np.tile(cluster_mean, (shape[0], 1))

    return tf.convert_to_tensor(adjusted_weights, dtype=tf.float32)

def create_custom_model(input_dim, output_dim, actual_values, he_init=True, mode='best_cluster'):
    model = models.Sequential()
    model.add(
        layers.Dense(
            64,
            activation='relu',
            kernel_initializer=lambda shape, dtype=None: clustered_initializer(shape, actual_values, he_init, mode=mode)
        )
    )

    model.add(
        layers.Dense(
            output_dim,
            activation='softmax',
            kernel_initializer=lambda shape, dtype=None: clustered_initializer(shape, actual_values, he_init, mode=mode)
        )
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

actual_values = y_train

# Model 1: He Initialization
model_he = create_custom_model(input_dim=784, output_dim=10, actual_values=actual_values, he_init=True)
start_time = time.time()
history_he = model_he.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=0)
he_time = time.time() - start_time

# Model 2: Xavier Initialization
model_xavier = create_custom_model(input_dim=784, output_dim=10, actual_values=actual_values, he_init=False)
start_time = time.time()
history_xavier = model_xavier.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=0)
xavier_time = time.time() - start_time

# Model 3: Clustered Weights (Best Cluster)
model_best_cluster = create_custom_model(input_dim=784, output_dim=10, actual_values=actual_values, mode='best_cluster')
start_time = time.time()
history_best_cluster = model_best_cluster.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=0)
best_cluster_time = time.time() - start_time

# Model 4: Averaged Weights
model_average = create_custom_model(input_dim=784, output_dim=10, actual_values=actual_values, mode='average')
start_time = time.time()
history_average = model_average.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=0)
average_time = time.time() - start_time

# Plot Results
plt.plot(history_he.history['loss'], label='He Initialization')
plt.plot(history_xavier.history['loss'], label='Xavier Initialization')
plt.plot(history_best_cluster.history['loss'], label='Best Cluster')
plt.plot(history_average.history['loss'], label='Average Weights')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history_he.history['val_accuracy'], label='He Initialization')
plt.plot(history_xavier.history['val_accuracy'], label='Xavier Initialization')
plt.plot(history_best_cluster.history['val_accuracy'], label='Best Cluster')
plt.plot(history_average.history['val_accuracy'], label='Average Weights')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Print Training Time
print(f"He Initialization Training Time: {he_time:.2f} seconds")
print(f"Xavier Initialization Training Time: {xavier_time:.2f} seconds")
print(f"Best Cluster Initialization Training Time: {best_cluster_time:.2f} seconds")
print(f"Average Weights Initialization Training Time: {average_time:.2f} seconds")

# ฟังก์ชันประเมินประสิทธิภาพ
def evaluate_model(model, x_test, y_test):
    # วัดเวลาที่ใช้ในการประเมินผล
    start_time = time.time()
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    inference_time = time.time() - start_time
    return accuracy, inference_time

# เก็บผลลัพธ์
results = []

# Training และวัดผลแต่ละเทคนิค
techniques = ['He Initialization', 'Xavier Initialization', 'Best Cluster', 'Average Weights']
models = [model_he, model_xavier, model_best_cluster, model_average]
training_times = [he_time, xavier_time, best_cluster_time, average_time]

for i, model in enumerate(models):
    print(f"Evaluating {techniques[i]}...")
    accuracy, inference_time = evaluate_model(model, x_test, y_test)
    results.append({
        'Technique': techniques[i],
        'Training Time (s)': training_times[i],
        'Inference Time (s)': inference_time,
        'Accuracy': accuracy
    })

# แสดงผลลัพธ์ในรูปแบบตาราง
print("\nPerformance Comparison:")
print("{:<25} {:<20} {:<20} {:<10}".format("Technique", "Training Time (s)", "Inference Time (s)", "Accuracy"))
for result in results:
    print("{:<25} {:<20.4f} {:<20.4f} {:<10.4f}".format(result['Technique'], result['Training Time (s)'], result['Inference Time (s)'], result['Accuracy']))
