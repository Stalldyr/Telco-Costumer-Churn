# Telco Customer Churn Analysis

## Overview
This project implements a machine learning solution for predicting customer churn in a telecommunications company. Using historical customer data, the system analyzes patterns and predicts which customers are likely to discontinue their services. The project includes comprehensive data exploration, feature engineering, and comparison of multiple machine learning models.

## Features
- **Data Exploration & Visualization**
  - Comprehensive EDA with automated visualization
  - Correlation analysis
  - Feature distribution analysis
  
- **Feature Engineering**
  - Creation of derived features (e.g., NumServices, AverageCharges)
  - Automated handling of different data types
  - Tenure-based feature creation
  
- **Machine Learning Models**
  - Random Forest
  - Logistic Regression
  - XGBoost
  - Cross-validation implementation

## Project Structure
- `Customer_Churn_Prediction.py`: Main script for data processing and model training
- `PlotBuilder.py`: Custom visualization class for creating various types of plots
- `ModelBuilder.py`: Machine learning model implementation and evaluation
- `column_types.json`: Configuration file for feature categorization
- `insights.md`: Presentations of the insights found in the data 
- `plots/`: Directory containing generated visualizations

## Package Versions
Package                       Version
----------------------------- ------------
matplotlib                    3.9.4
numpy                         1.24.3
pandas                        2.0.2
seaborn                       0.13.2
sklearn                       0.0.post9
xgboost                       2.1.3

## Data set

The data was acquired from Kaggle: 
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

