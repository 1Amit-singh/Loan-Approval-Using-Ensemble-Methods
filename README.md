# **Loan Approval Prediction System**

## 📜 **Overview**
The Loan Approval Prediction System is a web application leveraging machine learning to predict loan approval based on user inputs. It integrates a Flask backend, pre-trained ensemble machine learning models, and an interactive frontend to provide predictions and confidence scores.

This project showcases the application of data preprocessing, model ensemble techniques, and visualization tools in a real-world scenario, making it highly scalable and efficient.

---

## 🚀 **Features**
- Machine Learning-backed loan approval prediction.
- Ensemble modeling for robust and accurate results.
- Confidence score visualization using **Chart.js**.
- Fully responsive web interface.
- Scalable backend architecture.

---

## 🛠️ **Tech Stack**
| Component            | Technology         |
|----------------------|---------------------|
| **Frontend**         | HTML, CSS, JavaScript, Chart.js |
| **Backend**          | Python, Flask      |
| **Machine Learning** | Scikit-learn       |
| **Data Storage**     | Pickle (Serialized Models) |

---

## 📋 **How It Works**
1. **Input:**  
   User data is submitted through a form on the web interface.
2. **Data Preprocessing:**  
   The input data is preprocessed using a pre-fitted scaler to match the format expected by the machine learning models.
3. **Prediction:**  
   - An ensemble of models is used to predict the loan approval status.
   - Confidence scores (F1 scores) for each model are calculated using saved test data.
4. **Visualization:**  
   Model confidence scores are displayed as a bar chart for easy interpretation.

---

## 📁 **Directory Structure**
```plaintext
webapp/
├── templates/
│   ├── error.html               # Error page template
│   ├── index.html               # Input form
│   ├── result.html              # Prediction result page
├── app.py                       # Flask application
├── ensemble_model.pkl           # Serialized ensemble model
├── expected_columns.pkl         # Column names for input data
├── majority_voting_ensemble.py  # Ensemble model implementation
├── scaler.pkl                   # Pre-fitted scaler for input data
├── test_data.pkl                # Test data for F1 score calculation
├── loan_approval_dataset.csv    # Original dataset (optional)
├── loan_approval.ipynb          # Model training notebook
```
---

## ⚙️ **Setup Instructions**

### **Pre-requisites**
- Python 3.7 or above
- Pip for dependency management

### **Installation Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/loan-approval-prediction.git
   cd loan-approval-prediction/webapp
   ```

   
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```


3. Run Python File:
   ```bash
   python app.py
   ```

   The Project will run on http://127.0.0.1:5000

