# What Drives the Price of a Car?

This project explores a large dataset of used vehicles to identify the key factors that influence resale prices. The analysis is framed using the **CRISP-DM** methodology (Cross Industry Standard Process for Data Mining) and is designed to provide actionable insights for a used car dealership.

---

## 📌 Overview

- **Dataset**: Vehicles dataset (~426K rows, sampled from ~3M cars originally on Kaggle).
- **Goal**: Understand what makes a car more or less expensive.
- **Audience**: Used car dealers interested in fine-tuning inventory and pricing strategies.
- **Deliverable**: A report detailing the primary drivers of car prices and recommendations for inventory management.

---

## 🛠 Project Structure

- `notebook.ipynb` — Jupyter notebook containing the full analysis.
- `vehicles.csv` — Dataset of used cars (large file, may require sampling).
- `images/` — Supporting images (e.g., CRISP-DM diagram, Kurt image).
- `reports/` — Generated outputs (model performance, feature importance, client summary).

---

## 🔄 CRISP-DM Framework

1. **Business Understanding**  
   Identify dealership needs: which features drive car prices?

2. **Data Understanding**  
   Explore dataset, check distributions, missing values, and correlations.

3. **Data Preparation**  
   Clean target variable (`price`), impute missing values, engineer features, and normalize numeric data.

4. **Modeling**  
   Train regression models (Linear Regression, Ridge, Random Forest, Gradient Boosting).  
   Evaluate with cross-validation and holdout sets.

5. **Evaluation**  
   Assess model quality (R², RMSE). Identify top predictors using permutation importance and feature coefficients.

6. **Deployment**  
   Summarize findings in a client-facing report with actionable recommendations.

---

## 📊 Key Insights

- **Year**: Newer cars command higher prices.  
- **Mileage (Odometer)**: Lower mileage strongly increases value.  
- **Condition**: Better condition ratings correlate with higher resale prices.  
- **Manufacturer & Brand**: Premium brands yield higher margins; mass-market brands vary more.  
- **Vehicle Type**: SUVs and trucks generally price higher than sedans.  
- **Drive/Transmission**: AWD/4WD vehicles carry a premium.  
- **Fuel Type**: Hybrids and EVs can command higher prices if relatively new.

---

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Place `vehicles.csv` in the project directory.
3. Open `prompt_II.ipynb` in Jupyter Lab/Notebook.
4. Run cells sequentially:
   - Data loading & cleaning
   - Exploratory plots
   - Model training & evaluation
   - Client report generation

---

## 📈 Example Plots

- Distribution of car prices  
- Price vs. Year  
- Price vs. Odometer  
- Average price by manufacturer  
- Predicted vs. Actual prices scatterplot  
- Top features by importance

---

## 📌 Recommendations for Dealers

1. Source newer, low-mileage vehicles.  
2. Invest in reconditioning to improve condition ratings.  
3. Balance premium and mass-market brands.  
4. Highlight SUVs, trucks, and AWD/4WD vehicles.  
5. Stock newer hybrids/EVs for eco-conscious buyers.  

---

## ⚙️ Requirements# AIML-PracticalApp-WhatDrivesCarPrice
AI/ML Practical Application using Python - What Drives the Price of a Car? This is an assignment (Module 11) during my AI/ML course at UC Berkeley
