# Predictive Maintenance Application

## Project Title and Summary

This project aims to develop a Predictive Maintenance system using machine learning. The application predicts equipment failure based on sensor readings. It includes data preprocessing, model training, API development, and deployment.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [How to Run the Project](#how-to-run-the-project)
4. [File Structure](#file-structure)
5. [Core Functionality](#core-functionality)
6. [Testing](#testing)
7. [Deployment](#deployment)
8. [Challenges and Solutions](#challenges-and-solutions)
9. [Future Work](#future-work)

---

## Introduction

This project aims to develop a Predictive Maintenance system using machine learning. The application predicts equipment failure based on sensor readings. It includes data preprocessing, model training, API development, and deployment.

---

## Installation and Setup

- Python 3.10
- Libraries: pandas, scikit-learn, Keras, Flask

```
pip install -r requirements.txt
```

---

## How to Run the Project

```
python app.py
```

---

## File Structure

- data_preprocessing.py
- model_training.py
- app.py
- Dockerfile

---

## Core Functionality

### Data Preprocessing

1. Load raw sensor data.
2. Remove any missing or null values.
3. Normalize the features.

### Model Training

1. Split the data into training and test sets.
2. Train the model using a Neural Network.

### API Development

- `/predict`: POST request to get the prediction.

---

## Testing

- Unit tests for data preprocessing functions.
- Unit tests for model training functions.
- API endpoint tests.

---

## Deployment

The application is containerized using Docker and can be deployed on any cloud service that supports Docker containers.

```
docker build -t predictive-maintenance .
docker run -p 5001:5000 predictive-maintenance
```

---

## Challenges and Solutions

### Challenge: Data Imbalance

In many machine learning projects, especially in classification tasks, the dataset might be imbalanced. This means that the classes you are trying to predict are not represented equally. For example, in a binary classification task, you might have 90% of the samples in one class and only 10% in the other. This imbalance can introduce a bias towards the majority class, leading the machine learning model to make predictions skewed towards that class.

#### Issues with Data Imbalance:
- **Model Bias**: The model tends to favor the majority class, failing to capture the minority class's characteristics.
- **Misleading Metrics**: Accuracy can be misleading. A model that predicts only the majority class will have high accuracy but is not useful.
- **Overfitting**: There is a risk of overfitting for the minority class due to the lack of data.

### Solution: Using SMOTE for Oversampling

SMOTE (Synthetic Minority Over-sampling Technique) is a popular method for combating the data imbalance problem. It works by generating synthetic samples in the feature space. The algorithm takes a sample from the minority class and finds its k nearest neighbors. It then takes one of these neighbors and produces a new sample at a random point between the two in the feature space.

#### Steps to Apply SMOTE:
1. **Identify Imbalance**: First, identify the imbalance in your data using metrics like class distribution.
2. **Choose Relevant Metrics**: Choose performance metrics that are sensitive to data imbalance like F1-score, precision, recall, ROC-AUC, etc.
3. **Apply SMOTE**: Use SMOTE to oversample the minority class. It's important to only oversample the training data and leave the test data untouched.
4. **Train the Model**: Use the balanced data to train the machine learning model.
5. **Evaluate**: Evaluate the model using the untouched test data and the relevant metrics.

#### Benefits of SMOTE:
- **Better Generalization**: Because SMOTE generates synthetic examples, the model generalizes better to unseen data.
- **Improved Metrics**: Metrics like F1-score, recall, and precision often see significant improvement.

#### Cautions:
- **Over-generalization**: SMOTE can make the model too sensitive to the minority class, leading to an increase in false positives for the majority class.
- **Computationally Intensive**: The process can be computationally expensive for large datasets.

By applying SMOTE, the model's ability to identify the minority class improved significantly, thereby making the model more robust and reliable.

---

## Future Work

- Adding more machine learning models for comparison
- Implementing real-time sensor data streaming



