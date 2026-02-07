# Machine Learning From Scratch

A learning project implementing fundamental ML algorithms from scratch using NumPy.

## Overview

This repository contains implementations of classical machine learning algorithms built from the ground up without relying on high-level ML libraries. The goal is to understand the mathematical foundations and inner workings of these algorithms.

## Algorithms

- **K-Nearest Neighbors (KNN)** - Classification algorithm
- **Linear Regression** - Regression algorithm with gradient descent

## Inspiration

Based on tutorials from [AssemblyAI's YouTube channel](https://www.youtube.com/@AssemblyAI).

## Requirements

- Python 3.x
- NumPy
- scikit-learn (for data loading and evaluation)
- Matplotlib (for visualization)

## Setup

```bash
pip install numpy scikit-learn matplotlib
```

## Project Structure

```
├── KNN.py                  # K-Nearest Neighbors implementation
├── LinearRegression.py     # Linear Regression implementation
├── trainKNN.py             # Training script for KNN
├── trainLR.py              # Training script for Linear Regression
└── README.md
```

## Usage

Run the training scripts:

```bash
python train.py           # Test KNN on iris dataset
python trainLR.py         # Test Linear Regression
```

## Learning Goals

- Understand gradient descent optimization
- Implement distance metrics and neighbor voting
- Learn parameter tuning (learning rate vs iterations)
- Explore train/test split and model evaluation

---

*This is a learning project for educational purposes.*
