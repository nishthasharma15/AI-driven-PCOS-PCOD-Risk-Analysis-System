# AI-Driven PCOS/PCOD Risk Analysis System

## Overview

Polycystic Ovary Syndrome (PCOS) and Polycystic Ovarian Disease (PCOD) are common hormonal disorders affecting women of reproductive age. These conditions often remain undiagnosed due to overlapping symptoms, lack of early screening tools, and delayed medical consultation.

This project provides an **AI-driven risk prediction system** that uses machine learning models to analyze clinical and lifestyle parameters and predict the likelihood of PCOS/PCOD.

The goal is to assist in **early awareness and preliminary risk assessment** using data-driven methods.

---

## Problem Statement

Early diagnosis of PCOS/PCOD is challenging because:

- Symptoms vary widely between individuals  
- Multiple medical and lifestyle factors must be analyzed together  
- Many women do not have access to early screening support  

Therefore, the problem addressed is:

> To build a machine learning-based risk analysis system that predicts the possibility of PCOS/PCOD using patient health data.

---

## Objectives

- Clean and preprocess PCOS/PCOD dataset  
- Train machine learning classification models  
- Evaluate model performance  
- Provide a prediction system through a user interface  
- Generate architecture and evaluation visualizations  

---

## Key Features

- Complete dataset cleaning pipeline  
- Model training using Random Forest and baseline classifiers  
- StandardScaler integration for feature normalization  
- Saved trained models for direct deployment  
- Flask-based web application for prediction  
- Architecture diagrams and evaluation posters included  

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Flask (for deployment)  
- Pickle/Joblib (for saving models)  

---

## Project Structure

# File-by-File Explanation

## Dataset Files

### `PCOS_data.csv`
The **original raw dataset** containing clinical and lifestyle features such as:

- Age  
- BMI  
- Hormone levels  
- Menstrual irregularities  
- Follicle count  

Used as input for preprocessing and training.

---

### `PCOS_cleaned.csv`
The **cleaned dataset** generated after:

- Handling missing values  
- Removing inconsistencies  
- Formatting features properly  

Directly used for training machine learning models.

---

## Core Application File

### `app.py`
Main deployment file that:

- Loads the trained Random Forest model (`pcos_rf_model.pkl`)  
- Loads the scaler (`scaler.pkl`)  
- Accepts user health input through the interface  
- Predicts PCOS/PCOD risk  
- Displays the prediction result  

Run this file to start the application.

---

## Data Cleaning Module

### `clean.py`
Handles preprocessing steps such as:

- Loading raw dataset (`PCOS_data.csv`)  
- Cleaning missing or incorrect values  
- Saving the processed dataset as `PCOS_cleaned.csv`

---

## Model Training Files

### `model_basic_training.py`
Trains a **basic baseline machine learning model** for initial comparison.

Output model saved as:

- `pcos_basic_model.pkl`

---

### `model_training.py`
Main training pipeline that:

- Loads cleaned dataset  
- Splits data into training/testing sets  
- Applies feature scaling  
- Trains a Random Forest classifier  
- Evaluates performance  
- Saves the final model as:

- `pcos_rf_model.pkl`

---

## Model and Scaler Files

### `pcos_basic_model.pkl`
Serialized baseline trained model.

---

### `pcos_rf_model.pkl`
Final trained Random Forest classifier used for prediction.

---

### `scaler.pkl`
Saved StandardScaler object to normalize user input before prediction.

---

## Visualization File

### `visual.py`
Generates visual analysis such as:

- Correlation heatmaps  
- Feature importance plots  
- Model evaluation graphs  

Helps in understanding dataset patterns and model behavior.

---

## Architecture & Diagram Files

### `gen_architecture.py`
Script used to generate architectural workflow diagrams.

---

### `pcos_system_architecture.png`
System architecture diagram showing:

- Dataset input  
- Preprocessing  
- Model training  
- Prediction output  

---

### `pcos_system_architecture_poster.png`
Poster-style architecture diagram for presentations.

---

### `pcos_evaluation_poster.png`
Poster containing model evaluation results such as:

- Accuracy  
- Precision  
- Recall  
- Confusion Matrix  

---

## Model Used

### Random Forest Classifier

Chosen because:

- Works well on healthcare tabular datasets  
- Captures complex relationships between features  
- Provides high accuracy and feature importance  
- Reduces overfitting compared to single decision trees  

---
