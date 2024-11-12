import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
# ค่าเริ่มต้นของน้ำหนักและไบแอส
weights = {
    'w1': 0.15, 'w2': 0.20, 'w3': 0.25, 'w4': 0.30,
    'w5': 0.40, 'w6': 0.45, 'w7': 0.50, 'w8': 0.55
}
biases = {'b1': 0.35, 'b2': 0.60}

# ข้อมูลอินพุตและผลลัพธ์ที่ต้องการ
inputs = np.array([0.05, 0.10])
target_outputs = np.array([0.01, 0.99])

# ฟังก์ชัน Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ฟังก์ชันอนุพันธ์ของ Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# ฟังก์ชันคำนวณ feedforward
def feedforward(inputs, weights, biases):
    h1_input = inputs[0] * weights['w1'] + inputs[1] * weights['w3'] + biases['b1']
    h2_input = inputs[0] * weights['w2'] + inputs[1] * weights['w4'] + biases['b1']
    h1_output = sigmoid(h1_input)
    h2_output = sigmoid(h2_input)

    o1_input = h1_output * weights['w5'] + h2_output * weights['w7'] + biases['b2']
    o2_input = h1_output * weights['w6'] + h2_output * weights['w8'] + biases['b2']
    o1_output = sigmoid(o1_input)
    o2_output = sigmoid(o2_input)
    
    return np.array([o1_output, o2_output]), np.array([h1_output, h2_output])

# ฟังก์ชันคำนวณความผิดพลาด (Mean Squared Error)
def calculate_error(predicted, target):
    return np.mean((predicted - target) ** 2)

# การอัปเดตน้ำหนักด้วย Backpropagation
def backpropagation(weights, biases, inputs, hidden_outputs, outputs, target_outputs, learning_rate=0.01):
    output_errors = target_outputs - outputs
    output_deltas = output_errors * sigmoid_derivative(outputs)

    hidden_errors = np.dot(output_deltas, [weights['w5'], weights['w6']])
    hidden_deltas = hidden_errors * sigmoid_derivative(hidden_outputs)

    # ปรับน้ำหนักและไบแอส
    weights['w5'] += learning_rate * output_deltas[0] * hidden_outputs[0]
    weights['w6'] += learning_rate * output_deltas[0] * hidden_outputs[1]
    weights['w7'] += learning_rate * output_deltas[1] * hidden_outputs[0]
    weights['w8'] += learning_rate * output_deltas[1] * hidden_outputs[1]

    weights['w1'] += learning_rate * hidden_deltas[0] * inputs[0]
    weights['w2'] += learning_rate * hidden_deltas[0] * inputs[1]
    weights['w3'] += learning_rate * hidden_deltas[1] * inputs[0]
    weights['w4'] += learning_rate * hidden_deltas[1] * inputs[1]

    biases['b1'] += learning_rate * hidden_deltas.sum()
    biases['b2'] += learning_rate * output_deltas.sum()

    return weights, biases

# การอัปเดตน้ำหนักด้วยการผสาน Hill-climbing
def hill_climbing(weights, biases, inputs, target_outputs, iterations=1000, step_size=0.01):
    best_weights = weights.copy()
    best_biases = biases.copy()
    best_error = float('inf')

    for _ in range(iterations):
        # สุ่มปรับน้ำหนักเล็กน้อย
        new_weights = {k: v + step_size * (np.random.rand() - 0.5) for k, v in weights.items()}
        new_biases = {k: v + step_size * (np.random.rand() - 0.5) for k, v in biases.items()}
        
        outputs, _ = feedforward(inputs, new_weights, new_biases)
        error = calculate_error(outputs, target_outputs)

        if error < best_error:
            best_error = error
            best_weights = new_weights
            best_biases = new_biases

    return best_weights, best_biases

# การฝึกด้วยการเปรียบเทียบทั้งสองวิธีและการเก็บ Error ในแต่ละ Epoch
def train_and_compare():
    epochs = 10000
    learning_rate = 0.02

    # Train with Backpropagation only
    bp_weights = weights.copy()
    bp_biases = biases.copy()
    bp_errors = []

    for epoch in range(epochs):
        outputs, hidden_outputs = feedforward(inputs, bp_weights, bp_biases)
        bp_error = calculate_error(outputs, target_outputs)
        bp_errors.append(bp_error)
        bp_weights, bp_biases = backpropagation(bp_weights, bp_biases, inputs, hidden_outputs, outputs, target_outputs, learning_rate)

    # Backpropagation + Hill-climbing
    hc_weights, hc_biases = hill_climbing(weights.copy(), biases.copy(), inputs, target_outputs)
    hc_errors = []

    for epoch in range(epochs):
        outputs, hidden_outputs = feedforward(inputs, hc_weights, hc_biases)
        hc_error = calculate_error(outputs, target_outputs)
        hc_errors.append(hc_error)
        hc_weights, hc_biases = backpropagation(hc_weights, hc_biases, inputs, hidden_outputs, outputs, target_outputs, learning_rate)

    print("="*65)
    print("Error Backpropagation:", bp_errors[-1])
    print("Error Hill-climbing + Backpropagation:", hc_errors[-1])
    print("="*65)
    if bp_errors[-1] < hc_errors[-1]:
        print("Backpropagation win!!.")
    else:
        print("Hill-climbing + Backpropagation win!!")
    
    # Plotting the Errors for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(bp_errors, label="Backpropagation", color='blue')
    plt.plot(hc_errors, label="Hill-climbing + Backpropagation", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Comparison of Error Reduction: Backpropagation vs Hill-climbing + Backpropagation")
    plt.legend()
    plt.grid(True)
    plt.show()

# เรียกฟังก์ชัน train_and_compare() เพื่อเริ่มการฝึกและแสดงกราฟ
train_and_compare()
