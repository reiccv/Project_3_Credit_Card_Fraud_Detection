# Project 3 

## Credit Card Fraud Detection Machine Learning Model
This repository contains two different types of models that both aim for the same goal; detecting whether a transaction is fradulent or non fradulent. Containing a logistic Regression model and a neural network model for credit card fraud detection. 
The model is designed to analyze credit card transactions and predict whether a 
transaction is fraudulent or legitimate. It is trained on a labeled dataset of historical credit card transactions, 
where each transaction is labeled as either "fraud" or "non-fraud."

## Table of Contents

* [Installation Guide](##installation-guide)
* [Dataset](##dataset)
* [Neural Network Model](##neural-network-model)
* [Logistic Regression Model](##logistic-regression-model)
* [Model Evaluation Metrics](##model-evaluation-metrics)
* [Conclusions](#conclusions)

## Installation Guide
you need the following dependencies:

Python (>= 3.6)
TensorFlow (>= 2.0)
Keras (>= 2.4)
pandas (>= 1.1)
NumPy (>= 1.19)
You can install these dependencies using pip:

`pip install tensorflow keras pandas numpy`

### Imports

![]()

## Dataset

The dataset used to train and evaluate the model is included in this repository, you can find various publicly available datasets for credit card fraud detection on online platforms such as Kaggle or UCI Machine Learning Repository.

## Neural Network Model

Credit Card Fraud Detection Neural Network Model. Neural networks are capable of learning complex patterns and relationships in data. Credit card fraud detection requires identifying subtle and non-linear patterns in transaction data that might indicate fraudulent behavior. Neural networks excel at capturing such intricate patterns, making them well-suited for this task.

The neural network model is built using TensorFlow and Keras. It employs several layers, including input layers, hidden layers, and an output layer. The architecture is designed to handle the complexities of credit card transaction data and extract meaningful patterns for fraud detection.

### Data Cleaning

![]()

![]()

![]()

### High Accuracy 
When properly trained and optimized, neural network models can achieve high accuracy in detecting fraudulent transactions. 
They can significantly reduce false positives and false negatives, thus improving the overall performance of fraud detection systems.
Real time detection can also be acchieved through all of this with further advancements in the model.
In turn the model can keep learning and adapt to newer fraud patterns.

## Logistic Regression Model

## Model Evaluation Metrics
When evaluating the model's performance, consider the following metrics:

Accuracy: The overall proportion of correct predictions.
Precision: The proportion of true positive predictions out of all positive predictions, indicating the model's ability to avoid false alarms (incorrectly predicting fraud).
Recall (Sensitivity): The proportion of true positive predictions out of all actual positive cases, representing the model's ability to detect actual fraud cases.
F1 Score: The harmonic mean of precision and recall, providing a balanced evaluation of the model's performance.

## Conclusions

Credit card fraud detection is a critical and sensitive task. It is important to continually update and improve the model to stay ahead of new fraud patterns and techniques. Regularly retrain the model with the latest data and perform thorough evaluations to ensure its effectiveness. 

Credit card data should also be handled with the most care

Whats next for our model would be continously recalibrate it (retrain) and further research so the model can execute real time detection.