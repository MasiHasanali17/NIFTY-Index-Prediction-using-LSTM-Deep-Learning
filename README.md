```markdown
# NIFTY Index Prediction using LSTM Deep Learning

This project is an AI/ML-based deep learning application that predicts the next-day value of the NIFTY 50 index using historical Indian stock market data.  
It uses an LSTM (Long Short-Term Memory) neural network implemented in PyTorch to perform time-series forecasting.  
The project demonstrates how deep learning models can be applied to financial market analysis for educational purposes.

---

## 🚀 Features
- Predicts next-day NIFTY index value
- Uses LSTM deep learning model (PyTorch)
- Time-series forecasting on financial data
- Historical data fetched from Yahoo Finance
- Simple and clean implementation
- Optional Streamlit web interface

---

## 🧠 Technologies Used
- Python
- PyTorch
- LSTM (Long Short-Term Memory)
- Yahoo Finance API (`yfinance`)
- NumPy
- Pandas
- Scikit-learn
- Streamlit (for web interface)

---

## 📁 Project Structure
```

nifty-index-prediction-lstm/
├── app.py               # Streamlit application (UI version)
├── main.py              # Core training & prediction script
├── requirements.txt     # Python dependencies

````

---

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
````

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### 🔹 Option 1: Run with Streamlit (Recommended)

```bash
python -m streamlit run app.py
```

* Open the URL shown in the terminal (usually `http://localhost:8501`)
* The app will train the LSTM model and display the predicted next-day NIFTY value

---

### 🔹 Option 2: Run in Terminal (No UI)

```bash
python main.py
```

* Model trains in the terminal
* Prediction result is printed in the console

---

## 📊 Dataset

* Data Source: **Yahoo Finance**
* Ticker Used: **^NSEI (NIFTY 50 Index)**
* Time Period: Last 5 years
* Interval: Daily closing prices

---

## 🛠 Model Description

* Model Type: LSTM Neural Network
* Input: Last 30 days of closing prices
* Output: Next-day predicted index value
* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam

---

## 🛠 Remedy / Solution

The project provides a deep learning–based solution for forecasting the NIFTY index using historical time-series data.
By leveraging LSTM networks, the model captures long-term dependencies in market trends and produces next-day predictions.
This approach demonstrates how AI/ML techniques can be applied to financial data analysis in a practical and educational setting.

---

## 📌 Notes

* This project is intended for **educational purposes only**
* It does not provide financial or investment advice
* Predictions may not reflect real market behavior
* Focus is on learning deep learning and time-series forecasting

```
```
