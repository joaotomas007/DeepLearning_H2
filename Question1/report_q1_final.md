# Question 1: Image Classification with CNNs

## 1.1 Simple Convolutional Network

### Implementation Details
We implemented a Convolutional Neural Network (CNN) to classify images from the BloodMNIST dataset. The architecture follows the specifications:
-   **Conv Block 1**: 3 input channels $\to$ 32 output channels, kernel $3\times3$, stride 1, padding 1 + ReLU.
-   **Conv Block 2**: 32 $\to$ 64 output channels, kernel $3\times3$, stride 1, padding 1 + ReLU.
-   **Conv Block 3**: 64 $\to$ 128 output channels, kernel $3\times3$, stride 1, padding 1 + ReLU.
-   **Flatten**: Since no pooling was used, the spatial dimensions remained $28 \times 28$. The flattened feature vector has size $128 \times 28 \times 28 = 100,352$.
-   **Linear Layers**: A fully connected layer mapping $100,352 \to 256$ features (ReLU), followed by a final layer mapping $256 \to 8$ classes.

**Training Setup:**
-   **Optimizer**: Adam ($lr=0.001$)
-   **Loss Function**: `nn.CrossEntropyLoss`
-   **Epochs**: 200
-   **Batch Size**: 64

### Comparison: With vs. Without Softmax Layer
We conducted two experiments to verify the correct usage of the loss function:
1.  **Without Softmax (Logits)**: The model outputs raw scores. `nn.CrossEntropyLoss` applies `LogSoftmax` internally.
2.  **With Softmax**: The model applies `Softmax` before the loss function. This results in `LogSoftmax(Softmax(x))`.

**Results:**
| Metric | Without Softmax (Logits) | With Softmax |
| :--- | :--- | :--- |
| **Convergence** | Fast, stable, Loss $\to$ 0.0 | Stalled, Loss $\approx$ 1.5 |
| **Test Accuracy** | **~93.25%** | **~68.78%** |
| **Testing stability** | Stable | **Highly Unstable (Spikes)** |

**Discussion of Results and "Spikes":**
The model trained with the **Softmax layer** performed significantly worse (69% vs 93%).
-   **Optimization Failure**: The loss stalled at ~1.5 because the double application of Softmax restricts the inputs to the loss function to the interval [0,1]. The optimizer struggles to push the confidence higher because the gradients vanish or become distorted near the boundaries (0 and 1).
-   **The Spikes**: The "With Softmax" accuracy graph shows violent drops (spikes). This occurs because the optimizer builds momentum trying to force the Softmax output beyond 1.0 (an impossible target). When the weights shift slightly, the predictions collapse (e.g., flipping from one class to another or becoming uniform), causing the accuracy to plummet instantly before the model recovers.

---

## 1.2 Impact of MaxPool2d

### Implementation Changes
We modified the network by adding a `nn.MaxPool2d(kernel_size=2, stride=2)` layer after every ReLU activation in the convolutional blocks.

**Architecture Impact:**
-   **Dimensionality Reduction**: The spatial resolution is halved at each block ($28 \to 14 \to 7 \to 3$).
-   **Parameter Count**: The input to the first linear layer is reduced from $100,352$ (in 1.1) to **$1,152$** (in 1.2).
    -   Q1.1 Parameters (FC1): ~25.6 Million.
    -   Q1.2 Parameters (FC1): ~0.3 Million.

### Analysis of MaxPooling Impact

We repeated the experiments (Logits vs Softmax) with the new architecture.

**1. Effectiveness (Accuracy)**
-   **Logits (Best)**: Accuracy improved from **93.25%** (Q1.1) to **94.39%** (Q1.2).
-   The MaxPooling operation introduces **translation invariance** (small shifts in the input do not change the output) and forces the model to learn more robust, high-level features. It also acts as a regularizer, reducing overfitting.

**2. Efficiency (Training Time & Compute)**
-   **Training Time**: The unpooled model (Q1.1) took **~1210s** to train, while the pooled model (Q1.2) took only **~765s** (a **~37% reduction** in training time).
-   **Computational Cost**: Because the feature maps are smaller in deeper layers, the convolutional operations require fewer FLOPs. Crucially, the **98% reduction** in the first dense layer's weights drastically reduces memory usage and gradient computation time.

**3. Stability (Why did the spikes disappear?)**
In the Q1.2 "With Softmax" experiment, while the accuracy was still lower (~85.9%) than the Logits version, the violent spikes seen in Q1.1 disappeared.
-   **Reason**: The Q1.1 model had ~25 million parameters in the dense layer. In such a high-dimensional space, optimization instability (caused by the broken Softmax gradient) propagates explosively—there are millions of directions to "fall" into.
-   **Constrained Space**: The Q1.2 model has only ~300k parameters. This massive constraint stabilizes the optimization landscape. Even though the Double Softmax gradient is still "broken," the simpler model gets stuck in a stable local minimum rather than swinging wildly.

## Conclusion

1.  **Softmax vs Logits**: One should **never** apply a Softmax layer before `nn.CrossEntropyLoss`. It creates numerical instability, stalls convergence, and degrades final accuracy (from ~94% down to ~69-85%).
2.  **MaxPooling**: Adding MaxPooling layers is highly beneficial. It increased **Effectiveness** (+1.14% Accuracy) and radically improved **Efficiency** (faster training, 98% fewer parameters).
