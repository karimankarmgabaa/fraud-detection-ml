{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6d4bc5b4-bee3-4417-ade1-872162b81442",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction: Normal\n",
      "Fraud Probability: 0.1962605951379203\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "import pandas as pd\n",
    "\n",
    "# Load model & scaler\n",
    "model = joblib.load(\"fraud_model.pkl\")\n",
    "scaler = joblib.load(\"scaler.pkl\")\n",
    "\n",
    "# Load dataset (عشان نجيب مثال حقيقي)\n",
    "data = pd.read_csv(\"data/creditcard.csv\")\n",
    "\n",
    "# نفس preprocessing\n",
    "data[\"is_high_amount\"] = (data[\"Amount\"] > data[\"Amount\"].quantile(0.95)).astype(int)\n",
    "\n",
    "X = data.drop(\"Class\", axis=1)\n",
    "\n",
    "# خد sample واحدة\n",
    "X_new = X.iloc[[0]]  # أول صف\n",
    "\n",
    "# Scaling\n",
    "X_new_scaled = scaler.transform(X_new)\n",
    "\n",
    "# Prediction\n",
    "probs = model.predict_proba(X_new_scaled)\n",
    "\n",
    "threshold = 0.98\n",
    "prediction = (probs[:,1] >= threshold).astype(int)\n",
    "\n",
    "print(\"Prediction:\", \"Fraud\" if prediction[0]==1 else \"Normal\")\n",
    "print(\"Fraud Probability:\", probs[0][1])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
