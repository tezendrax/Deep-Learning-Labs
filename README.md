# 🧠 Deep Learning Labs

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Welcome to the **Deep Learning Labs** repository! This repository contains a curated collection of hands-on lab experiments, model implementations from scratch, neural network architectures, and deep learning projects using **PyTorch**, **TensorFlow/Keras**, and **Python**.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Lab Modules](#-lab-modules)
  - [Lab 1: Foundations & Multi-Layer Perceptrons (MLP)](#lab-1-foundations--multi-layer-perceptrons-mlp)
  - [Lab 2: Optimization Algorithms & Regularization](#lab-2-optimization-algorithms--regularization)
  - [Lab 3: Convolutional Neural Networks (CNNs)](#lab-3-convolutional-neural-networks-cnns)
  - [Lab 4: Recurrent Neural Networks & Sequence Modeling](#lab-4-recurrent-neural-networks--sequence-modeling)
  - [Lab 5: Autoencoders & Representation Learning](#lab-5-autoencoders--representation-learning)
  - [Lab 6: Generative Adversarial Networks (GANs)](#lab-6-generative-adversarial-networks-gans)
  - [Lab 7: Attention Mechanisms & Transformers](#lab-7-attention-mechanisms--transformers)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Getting Started](#-getting-started)
- [Visualizations & Results](#-visualizations--results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔬 Overview

This repository serves as a comprehensive practical guide for understanding core concepts and state-of-the-art techniques in Deep Learning. Each lab includes:
- 📖 **Theoretical Notes & Mathematical Foundations**
- 💻 **Clean, Well-Documented Code (Jupyter Notebooks & Python Scripts)**
- 📊 **Model Training Logs, Loss Curves & Evaluation Metrics**
- 🚀 **Interactive Google Colab Notebooks**

---

## 📂 Repository Structure

```text
Deep-Learning-Labs/
│
├── 01-MLP-Foundations/
│   ├── perceptron_from_scratch.py
│   ├── mlp_forward_backward_propagation.ipynb
│   └── activation_functions_visualized.ipynb
│
├── 02-Optimization-Regularization/
│   ├── gradient_descent_variants.ipynb
│   ├── dropout_and_batchnorm.ipynb
│   └── weight_initialization.ipynb
│
├── 03-CNN-Image-Classification/
│   ├── cnn_from_scratch_mnist.ipynb
│   ├── transfer_learning_resnet_vgg.ipynb
│   └── data_augmentation_techniques.ipynb
│
├── 04-RNN-LSTM-Sequence-Models/
│   ├── rnn_text_generation.ipynb
│   ├── lstm_time_series_forecasting.ipynb
│   └── gru_sentiment_analysis.ipynb
│
├── 05-Autoencoders-VAEs/
│   ├── vanilla_convolutional_autoencoders.ipynb
│   └── variational_autoencoder_vae.ipynb
│
├── 06-GANs-Generative-Models/
│   ├── vanilla_gan_mnist.ipynb
│   └── dcgan_image_generation.ipynb
│
├── 07-Transformers-Attention/
│   ├── self_attention_from_scratch.ipynb
│   └── fine_tuning_bert_huggingface.ipynb
│
├── datasets/            # Sample datasets & data download scripts
├── utils/               # Helper scripts for plotting and metrics
├── requirements.txt     # Python dependencies
└── README.md            # Repository overview
```

---

## 🧪 Lab Modules

### Lab 1: Foundations & Multi-Layer Perceptrons (MLP)
- **Topics**: Perceptron learning algorithm, activation functions (Sigmoid, Tanh, ReLU, Softmax), forward pass, backpropagation algorithm, cross-entropy loss.
- **Key Task**: Implementing a complete Multi-Layer Perceptron from scratch using NumPy.

### Lab 2: Optimization Algorithms & Regularization
- **Topics**: SGD, Momentum, RMSprop, Adam optimizer; L1/L2 regularization (weight decay), Dropout layers, Batch Normalization, Learning Rate Schedules.
- **Key Task**: Comparing optimization convergence speed and combating overfitting on noisy data.

### Lab 3: Convolutional Neural Networks (CNNs)
- **Topics**: Convolution operations, kernel filters, pooling (Max/Avg), stride, padding, receptive fields, popular architectures (LeNet-5, AlexNet, VGG16, ResNet50).
- **Key Task**: Image classification on CIFAR-10/MNIST and applying Transfer Learning.

### Lab 4: Recurrent Neural Networks & Sequence Modeling
- **Topics**: Sequential data processing, Vanishing/Exploding Gradient Problem, Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), Bidirectional RNNs.
- **Key Task**: Time-series forecasting and sequence-to-sequence text generation.

### Lab 5: Autoencoders & Representation Learning
- **Topics**: Dimensionality reduction, feature extraction, Denoising Autoencoders, Convolutional Autoencoders, Variational Autoencoders (VAEs), KL Divergence.
- **Key Task**: Image reconstruction, noise removal, and latent space visualization.

### Lab 6: Generative Adversarial Networks (GANs)
- **Topics**: Minimax game formulation, Generator & Discriminator network dynamics, Deep Convolutional GANs (DCGAN), Wasserstein GAN (WGAN).
- **Key Task**: Synthetic image generation from random noise vectors.

### Lab 7: Attention Mechanisms & Transformers
- **Topics**: Scaled Dot-Product Attention, Multi-Head Self-Attention, Positional Encodings, Transformer Encoder & Decoder architectures, BERT & Vision Transformers (ViT).
- **Key Task**: Implementing self-attention from scratch and fine-tuning a pretrained Transformer.

---

## 🛠️ Prerequisites & Installation

### Prerequisites
- Python `3.10` or higher
- `pip` package manager
- (Optional) CUDA-enabled GPU for accelerated training

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tezendrax/Deep-Learning-Labs.git
   cd Deep-Learning-Labs
   ```

2. **Create and activate a virtual environment**:
   - **On Linux/macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **On Windows**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Getting Started

Launch Jupyter Notebook or JupyterLab to explore the labs:

```bash
jupyter notebook
```

Alternatively, you can open any notebook directly in **Google Colab** by clicking the badges provided inside each lab folder.

---

## 📊 Requirements (`requirements.txt`)

Key libraries used across the labs:
- `torch` / `torchvision`
- `tensorflow` / `keras`
- `numpy` & `pandas`
- `matplotlib` & `seaborn`
- `scikit-learn`
- `jupyter` / `notebook`
- `transformers` & `datasets`

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are always welcome! Feel free to:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingLab`)
3. Commit your changes (`git commit -m 'Add new lab experiment'`)
4. Push to the branch (`git push origin feature/AmazingLab`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

<p align="center">
  Crafted with ❤️ by <a href="https://github.com/tezendrax">Tejendra Singh</a>
</p>
