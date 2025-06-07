# Maternal-health-risk-prediction-
# Maternal Health Risk Predictor

This is a simple machine learning project I did. The goal is to predict the risk level of maternal health (low, mid, or high) using basic health data like blood pressure, blood sugar, heart rate, and temperature.

I chose this topic because maternal health is a serious issue in places like Africa, where early risk detection can be the difference between life and death. The dataset isn’t from Africa
, but the features are universal, so the model could still be useful with local data in the future.

---

## Dataset

I used a small dataset from Kaggle:  
https://www.kaggle.com/datasets/andrewmvd/maternal-health-risk-data

It includes 73 samples with the following columns:
- Age
- SystolicBP
- DiastolicBP
- BS (Blood Sugar)
- BodyTemp
- HeartRate
- RiskLevel (target)

---

## What the Project Does

- Loads and cleans the data
- Visualizes feature distributions and risk levels
- Encodes the target and scales the features
- Trains a logistic regression model
- Evaluates accuracy and other metrics
- Shows the results using visualizations like confusion matrix and classification scores

---

## How to Run

1. Clone the repo or download the files
2. Make sure the CSV file is in the same folder
3. Run the Python script:

```bash
python3 maternal_health_predictor.py

