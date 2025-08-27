# Mini Transformer from Scratch in Rust

A fully functional mini Transformer block implemented **from scratch in Rust**, including:

- Multi-head attention (single-head prototype)  
- Feed-forward network (FFN)  
- Residual connections  
- Softmax cross-entropy loss  
- Manual backpropagation (autograd)  

This project is a hands-on exploration of **how Transformers work at a low level**, without relying on high-level ML libraries like PyTorch or TensorFlow. It’s a great resource for anyone who wants to understand **attention mechanisms, gradient flow, and neural network training** from first principles.

---

## Features

- ✅ `TensorNode` struct for storing values and gradients  
- ✅ Forward pass for attention, FFN, residual connections  
- ✅ Backward pass with proper gradient accumulation  
- ✅ Softmax and cross-entropy loss  
- ✅ Fully functional example training step with gradient printout  

---

## Mathematical Overview

### Forward Pass

1. **Linear projections for attention:**  
$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$

2. **Scaled dot-product attention:**  
$S = Q K^\top, \quad A = \text{softmax}(S), \quad O = A V$

3. **Residual connection:**  
$R_1 = X + O$

4. **Feed-forward network:**  
$F_1 = R_1 W_1, \quad H = \text{ReLU}(F_1), \quad F_2 = H W_2$

5. **Second residual:**  
$R_2 = R_1 + F_2$

6. **Loss:**  
$L = \text{cross entropy}(R_2, targets)$

---

### Backward Pass

Gradient flow follows the chain rule:

$\nabla_X = \nabla_X^{residual} + \nabla_X^{attention}$

$\nabla_{W_Q} = X^\top \nabla_Q, \quad \nabla_{W_K} = X^\top \nabla_K, \quad \nabla_{W_V} = X^\top \nabla_V$

…and similarly for $W_1$ and $W_2$.

---

## Getting Started

### Prerequisites

- Rust 1.70+  
- Cargo (comes with Rust)

### Clone the repository

```
git clone https://github.com/AbdullaShafeeg/miniTransformer.git
cd miniTransformer
```

## Getting Started

### Run the Project

```
cargo run
```

## Project Structure

- **TensorNode**: Struct storing tensor values and gradients  
- **multi_head_attention_forward/backward**: Attention computations  
- **feed_forward_forward/backward**: FFN computations  
- **softmax_cross_entropy_loss/backward**: Loss and gradient computation  

---

## Learning Outcomes

- Understand how **attention and feed-forward layers** compute outputs  
- Manually implement **backpropagation for matrix operations**  
- Gain insight into **residual connections and gradient accumulation**  
- Learn how **softmax and cross-entropy** interact during training  

---

## License

MIT License © 2025 Abdulla Shafeeg
