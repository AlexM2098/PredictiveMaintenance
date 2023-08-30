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
10. [Acknowledgments](#acknowledgments)
11. [License](#license)

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

- Challenge: Data imbalance
  - Solution: Used SMOTE for oversampling

---

## Future Work

- Adding more machine learning models for comparison
- Implementing real-time sensor data streaming



