# Weather Prediction Using SVR

## Author
**Mohammad Yasin Nur Akib**  
Computer Science & Technology  
Nanjing University of Information Science & Technology  

---

## 📌 Project Overview
This project focuses on predicting precipitation using **Support Vector Regression (SVR)** based on meteorological data.  
It involves preprocessing weather variables, creating lagged features, training an SVR model, and evaluating prediction performance through statistical metrics and visualizations.

---

## 🎯 Key Features
- **SVR Model** for precipitation prediction
- **Time-series data preprocessing**
- **Lagged feature engineering** for temperature and wind speed
- **Model evaluation** using MSE and R² score
- **Visualization** of results (time-series, residual plots, feature importance)

---

## 🗂 Dataset Information (Important)
This project uses **meteorological NetCDF (`.nc`) data** (e.g., ERA5 reanalysis data).

⚠️ **The `.nc` dataset files are NOT included in this repository** due to GitHub file size limitations and best practices.

### How to obtain the dataset:
- Download the required NetCDF data from an official source such as:
  - https://cds.climate.copernicus.eu/
- After downloading, place the `.nc` files in a local folder (e.g. `Data/`) as referenced in the code.

📌 The repository contains **only the code**, ensuring the project remains lightweight and reproducible.

---

## 🔧 Installation & Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
pip install -r Requirments.txt
python Akib.py
