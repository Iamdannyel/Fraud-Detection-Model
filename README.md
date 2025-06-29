### Fraud Detection with PaySim Dataset


### Overview


This project develops a machine learning model to detect fraudulent transactions using the PaySim synthetic dataset, which simulates mobile money transactions. The dataset contains 6.3M transactions with a ~0.13% fraud rate, subsampled to ~193,097 rows (1.28% fraud) for efficiency. The goal is to identify fraud while addressing challenges like class imbalance, data leakage, and feature engineering.

XGBoost is used to achieve realistic performance metrics (precision ~0.85, recall ~0.80, F1-score ~0.82) after resolving issues such as data leakage from features like is_balance_drained (correlation ~0.989 with isFraud).
Features

Numerical: oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, hour, log_amount, step, amount.
Categorical: type (encoded as type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER).
Target: isFraud (1 for fraud, 0 for non-fraud).
Dropped: nameOrig, nameDest (identifiers).
Excluded (due to leakage): is_balance_drained, tx_type_C-C, tx_type_C-M.

Project Structure
fraud-detection-paysim/
├── data/
│   ├── raw/                    # Raw PaySim dataset (not included, download from Kaggle)
│   └── processed/              # Processed data (e.g., subsampled CSV)
├── notebooks/                  # Jupyter notebooks for EDA and experiments
│   └── Fraud_Detection.ipynb
├── output/                     # Model outputs, logs, and metrics
│   ├── fraud_detection.log
│   └── test_metrics.txt
├── src/                        # Python scripts
│   └── train_model.py
├── README.md                   # This file
├── requirements.txt            # Dependencies
└── .gitignore                  # Git ignore file

Prerequisites

Python: 3.8+
Dependencies:pip install -r requirements.txt

See requirements.txt:pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0


Dataset: Download data from https://www.kaggle.com/aiyazmiran/paysim-part to transactions_data/raw/.

Setup

Clone the repository:git clone https://github.com/your-username/fraud-detection-paysim.git
cd fraud-detection-paysim


Install dependencies:pip install -r requirements.txt


Prepare data:
Place synthetic_financial_datasets.csv in data/raw/.
Run preprocessing to subsample and clean data (see notebooks/Fraud_Detection.ipynb).



Usage

Exploratory Data Analysis (EDA):
Open notebooks/Fraud_Detection.ipynb to explore data insights (e.g., fraud in CASH_OUT/0.06% fraud rate, TRANSFER/0.77% fraud rate, ~60–65% fraud with newbalanceOrig = 0).


Train the model:
Run the training script:python src/train_model.py


This trains a Random Forest model, saves metrics in output/test_metrics.txt, and logs in output/fraud_detection.log.


Evaluate results:
Check output/test_metrics.txt for metrics (e.g., precision ~0.85, recall ~0.80, F1-score ~0.82).
Review output/fraud_detection.log for training details.



Key Findings

Initial Issues:
Data leakage from is_balance_drained (correlation ~0.989 with isFraud) caused unrealistic metrics (precision/recall ~1.00).
Missing numerical features led to poor performance (precision ~0.003, high false positives).


Resolution:
Excluded leaky features (is_balance_drained, tx_type_C-C, tx_type_C-M).
Included numerical features (newbalanceOrig, log_amount) and properly encoded type.
Achieved realistic metrics: precision ~0.85, recall ~0.80, F1-score ~0.82.


Challenges:
Class imbalance (~1.28% fraud in subsample) addressed with class_weight='balanced'.
Overly discriminative subsamples amplified leakage risks.



Performance

Training Time: ~20–30s for ~135,168 rows (subsample).
Testing Time: ~5s for ~57,929 rows.
Memory: ~600MB for subsample; ~10GB for full ~6.3M rows.
Metrics (Random Forest, test set):Accuracy: 0.9997
Confusion Matrix:
[[190629      4]
 [    492   1972]]
Classification Report:
              precision    recall  f1-score   support
         0       1.00      1.00      1.00    190633
         1       0.85      0.80      0.82      2464



Future Improvements

Experiment with additional models (e.g., LightGBM, neural networks).
Engineer new features (e.g., balance_error_orig, time_of_day).

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-name).
Open a pull request.

License
This project is licensed under the MIT License. See LICENSE for details.
Contact
For questions or feedback, contact [your-email@example.com] or open an issue on GitHub.
