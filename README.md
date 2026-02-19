I developed an end-to-end Machine Learning regression project to predict medical insurance charges based on demographic and health-related factors such as age, BMI, smoking status, and region.

I performed data preprocessing using Scikit-learn's ColumnTransformer and Pipeline, trained a RandomForestRegressor model, evaluated it using MAE, RMSE, and R² score, and deployed the trained model using FastAPI to provide real-time predictions via REST API.
🏥 Medical Insurance Cost Prediction API
📌 Project Overview

This project predicts medical insurance charges based on customer information such as:

Age

Gender

BMI

Number of children

Smoking status

Region

The model is built using Machine Learning (Regression) and deployed using FastAPI to provide real-time predictions.

🎯 Problem Statement

Insurance companies need to estimate medical charges based on customer risk factors.
This project builds a regression model to predict insurance costs accurately.

Target Variable: charges

🛠 Tech Stack

Python

Pandas

NumPy

Scikit-learn

FastAPI

Uvicorn

Joblib
📊 Machine Learning Details

Problem Type: Regression

Algorithm Used: RandomForestRegressor

Preprocessing:

OneHotEncoding for categorical features

ColumnTransformer

Pipeline integration

Evaluation Metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score
