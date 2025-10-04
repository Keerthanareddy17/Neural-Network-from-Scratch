# Building a Neural Network from Scratch!

![math-numpy-python](https://github.com/user-attachments/assets/1ba4957d-1209-442f-8da9-1ff367cf10ca)

This project was all about understanding *how* a neural network actually works....... **from math to code**.  
No TensorFlow, no PyTorch, just **NumPy** and some core linear algebra.  
I used the **[MNIST](https://www.kaggle.com/competitions/exploretechla-ignite-digit-recognition-challenge/data) handwritten digit dataset** and trained a small neural network to classify digits from `0–9`.

---

## Dataset

Each row in the CSV file represents one 28×28 grayscale image of a digit.

| Column | Description |
|--------|--------------|
| `label` | The digit (0–9) |
| `pixel0`–`pixel783` | Flattened 28×28 image pixels |

Sample:

| label | pixel0 | pixel1 | ... | pixel782 | pixel783 |
|--------|---------|--------|------|------------|-----------|
| 5 | 0 | 0 | ... | 0 | 0 |
| 3 | 0 | 0 | ... | 0 | 0 |

---

## Step 1: Data Preparation

- We first load and shuffle the dataset
- Then we split it into training and validation sets
- Finally, we normalize pixel values from [0, 255] → [0, 1] for stable gradients:
```python
X_train = X_train / 255.0
X_val = X_val / 255.0

```
---
## Step 2: Neural Network Architecture

Our architecture is intentionally simple:
```
Input Layer (784) → Hidden Layer (10 neurons, ReLU) → Output Layer (10 neurons, Softmax)
```
- Input layer (784): one neuron per pixel (28×28)
- Hidden layer (10 neurons): learns simple features
    - Activation: ReLU -> *f*(z)=max(0,z)
- Output layer (10 neurons): represents digits 0–9
    - Activation: Softmax
 
      <img width="782" height="488" alt="Screenshot 2025-10-04 at 10 31 33" src="https://github.com/user-attachments/assets/ff6b5944-36e2-45e8-a0e3-16a6bfc8be4b" />

 
---

## Step 3: Initializing Parameters

We initialize small random weights and biases: 
```python
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

```

## Step 4: Forward Propagation

The Math :
- *Z1 ​= W1​X + b1*
- *A1​ = ReLU(Z1​)*
- *Z2​ = W2​A1 ​+ b2​​*
- *A2​ = Softmax(Z2​)*

---

## Step 5: Backpropagation

We calculate the gradients of the loss function with respect to `weights` and `biases`.

- First, we convert labels into one-hot form.
  ```python
  def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
  ```
- Then compute errors and propagate them backward.

  ```python
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
  ```
  ---

## Step 6: Parameter Updates

Using gradient descent:
    - *W = W − α⋅dW*
    - *b = b − α⋅db*      (where *α* = learning rate)
```python
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
  ```
---

## Step 7: Training (Gradient Descent Loop)

We train the model for `500` iterations:

During training, the accuracy gradually improves from `~10%` -> `83%`+ on training data


---

## Step 8: Testing on Validation Data
```python
    def evaluate(X, Y, W1, b1, W2, b2):
        predictions = make_predictions(X, W1, b1, W2, b2)
        return get_accuracy(predictions, Y)
    
    print("Accuracy:", evaluate(X_val, Y_val, W1, b1, W2, b2))
  ```

Validation Accuracy: `~84.5%`

<img width="184" height="39" alt="Screenshot 2025-10-04 at 11 01 39" src="https://github.com/user-attachments/assets/19661ff9-c633-40fb-9c9e-a6c9c475ef85" />

---

#### Final Thoughts
- This project was a fun way to brush up the basics again....... 
- While frameworks like PyTorch or TensorFlow handle all this automatically, implementing it manually helps you really understand:
    - How forward and backward propagation work
    - How gradients shape the learning process
    - How the math connects to actual prediction results
- Achieved: ~84.5% accuracy on MNIST using just NumPy 

*Built purely for learning.......because understanding what’s under the hood is always worth it :)*
