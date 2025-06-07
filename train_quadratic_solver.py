import math
import random
import json


def generate_equation(min_val=-10.0, max_val=10.0):
    """Generate coefficients a, b, c and their real roots."""
    while True:
        a = random.uniform(min_val, max_val)
        if abs(a) < 1e-6:
            continue
        b = random.uniform(min_val, max_val)
        c = random.uniform(min_val, max_val)
        disc = b * b - 4 * a * c
        if disc < 0:
            continue
        sqrt_disc = math.sqrt(disc)
        x1 = (-b + sqrt_disc) / (2 * a)
        x2 = (-b - sqrt_disc) / (2 * a)
        return a, b, c, x1, x2


def relu(x):
    return [max(0.0, v) for v in x]


def drelu(x):
    return [1.0 if v > 0 else 0.0 for v in x]


def dot(x, w_col):
    return sum(x_i * w_i for x_i, w_i in zip(x, w_col))


def forward(x, params):
    W1, b1, W2, b2 = params
    # Hidden layer
    h = [dot(x, [W1[i][j] for i in range(len(x))]) + b1[j] for j in range(len(b1))]
    h_act = relu(h)
    # Output layer
    y = [dot(h_act, [W2[j][k] for j in range(len(h_act))]) + b2[k] for k in range(len(b2))]
    return h_act, y


def backward(x, h_act, y_pred, y_true, params, lr):
    W1, b1, W2, b2 = params
    dloss_dy = [2 * (y_pred[k] - y_true[k]) for k in range(2)]
    # Update W2 and b2
    for j in range(len(h_act)):
        for k in range(2):
            W2[j][k] -= lr * dloss_dy[k] * h_act[j]
    for k in range(2):
        b2[k] -= lr * dloss_dy[k]
    # Gradients through ReLU
    dh = [sum(dloss_dy[k] * W2[j][k] for k in range(2)) * (1.0 if h_act[j] > 0 else 0.0) for j in range(len(h_act))]
    # Update W1 and b1
    for i in range(len(x)):
        for j in range(len(h_act)):
            W1[i][j] -= lr * dh[j] * x[i]
    for j in range(len(h_act)):
        b1[j] -= lr * dh[j]


def initialize_network(input_size=3, hidden_size=16, output_size=2):
    random.seed(0)
    W1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
    b1 = [0.0 for _ in range(hidden_size)]
    W2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
    b2 = [0.0 for _ in range(output_size)]
    return [W1, b1, W2, b2]


def train(iterations=1_000_000, lr=1e-3):
    params = initialize_network()
    for step in range(iterations):
        a, b, c, x1, x2 = generate_equation()
        x = [a, b, c]
        h_act, y_pred = forward(x, params)
        backward(x, h_act, y_pred, [x1, x2], params, lr)
        if (step + 1) % 10000 == 0:
            loss = sum((y_pred[i] - [x1, x2][i]) ** 2 for i in range(2))
            print(f"Step {step + 1}: loss={loss:.6f}")
    return params


def save_model(params, path="quadratic_model.json"):
    with open(path, "w") as f:
        json.dump(params, f)


if __name__ == "__main__":
    model_params = train()
    save_model(model_params)
