🌸 Iris Classifier with Explainability
An advanced Iris Flower Classification project built using Random Forest, featuring Hyperparameter Tuning (GridSearchCV), Exploratory Data Analysis (EDA), Model Evaluation, and Explainable AI (SHAP). Includes multiple visualizations and saves plots for reporting.

📌 Features
✔ Load and preprocess Iris dataset
✔ Perform EDA with Seaborn and Matplotlib
✔ Train Random Forest Classifier with GridSearchCV for best hyperparameters
✔ Evaluate model with:

Confusion Matrix (absolute & normalized)

Classification Report

ROC Curves with AUC (multi-class)
✔ Feature Importance Visualization
✔ Explainability with SHAP values
✔ Auto-save all plots for analysis

🗂 Project Structure
bash
Copy
Edit
🔧 Installation
Make sure you have Python 3.8+ installed.

1. Clone this repository
bash
Copy
Edit
git clone https://github.com/your-username/iris-classifier.git
cd iris-classifier
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Or install manually:

bash
Copy
Edit
pip install pandas seaborn matplotlib numpy scikit-learn shap
▶️ How to Run
bash
Copy
Edit
python app.py
All plots will be displayed and saved in the project directory.

📊 Visualizations Generated
Class distribution bar chart

Correlation heatmap

Pairplot (feature relationships)

Confusion matrix (absolute & normalized)

ROC-AUC curves for all classes

Feature importance chart

SHAP summary plot for model interpretability

🧠 Model Details
Algorithm: Random Forest Classifier

Hyperparameter Tuning: GridSearchCV (5-fold cross-validation)

Metrics: Accuracy, Classification Report, ROC-AUC

✅ Best Parameters Example
python
Copy
Edit
{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
🔍 Explainability
This project uses SHAP (SHapley Additive exPlanations) to explain individual predictions and global feature importance.

📌 Requirements
Python 3.8+

pandas

seaborn

matplotlib

numpy

scikit-learn

shap










Ask ChatGPT
