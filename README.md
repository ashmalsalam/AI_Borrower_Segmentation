# AI-Based Borrower Segmentation & Default Prediction for Debt Recovery
 
**Author:** Ashmal Abdussalam P T

**Dataset:** UCI Default of Credit Card Clients — 30,000 real records  
**Objective:** Segment borrowers by risk profile using unsupervised learning and predict default probability using supervised ML — enabling personalized, AI-driven debt recovery strategies.
 
---
 
## 📊 Results
 
| Model | CV AUC (mean) | Test AUC | Notes |
|---|---|---|---|
| Logistic Regression | 0.7577 ± 0.0058 | 0.7423 | Baseline linear model |
| Random Forest | 0.7829 ± 0.0059 | 0.7741 | Ensemble, handles non-linearity |
| **XGBoost** | **0.7835 ± 0.0068** | **0.7749** | Top performer |
 
 
**Top predictive features:** `PAY_0` (most recent payment status), `payment_consistency`, `utilization_rate`, `max_delay`
 
---
 
## 🗂️ Borrower Segments
 
| Segment | Default Rate | Recovery Strategy |
|---|---|---|
| **Low Risk** | ~10–12% | Soft AI nudges — payment reminders, positive reinforcement |
| **Medium Risk** | ~20–25% | Flexible EMI restructuring, AI-negotiated repayment plans |
| **High Risk** | ~40–50%+ | Escalated AI intervention, intensive counselling, legal triggers |
 
---
 
## 🔬 Pipeline Overview
 
1. **Data Loading & EDA** — distributions, class imbalance, default rate by payment status, correlation heatmap
2. **Feature Engineering** — 7 new financial features built from raw data:
   - `utilization_rate` — credit used vs credit limit
   - `repayment_ratio` — actual payment vs billed amount
   - `payment_consistency` — months paid on time in last 6 months
   - `max_delay` — worst payment delay across 6 months
   - `avg_payment_3m` / `avg_bill_3m` — rolling 3-month averages
   - `debt_to_payment` — bill vs payment ratio (debt pressure proxy)
3. **Borrower Segmentation (KMeans)** — elbow method + silhouette score to determine optimal k=3, clusters auto-labelled Low/Medium/High Risk by actual default rate
4. **Default Prediction** — Logistic Regression, Random Forest, XGBoost with StratifiedKFold CV and class imbalance handling
5. **Model Evaluation** — ROC-AUC curves, confusion matrix, classification report
6. **Feature Importance** — XGBoost built-in + permutation importance (model-agnostic)
7. **Business Strategy Layer** — each segment mapped to a Riverline-style AI recovery strategy
 
---
 
## 🗃️ Dataset
 
**UCI Default of Credit Card Clients**
- 30,000 real credit card records from Taiwan
- 23 features including payment history, bill amounts, demographic info
- Target: `default payment next month` (binary)
- Default rate: ~22% (class imbalance handled with `class_weight='balanced'` and `scale_pos_weight`)
 
Download: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
 
---
 
## 🛠️ Tech Stack
 
`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Matplotlib` `Seaborn`
 
---
 
## 📂 Repository Structure
 
```
ai-borrower-segmentation/
│
├── AI_Borrower_Segmentation_Riverline.ipynb 
├── README.md
```
 
---
 
## 🚀 How to Run
 
```bash
# Clone the repo
git clone https://github.com/ashmalsalam/ai-borrower-segmentation.git
cd ai-borrower-segmentation

# Download the dataset from https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
 
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl jupyter
 
# Run the notebook
jupyter notebook AI_Borrower_Segmentation_Riverline.ipynb
```
  
---
 
## 💡 Business Application
 
This project directly maps to how an AI-first debt recovery platform (like Riverline AI) could:
 
- **Triage incoming borrowers** — assign a segment + default probability score at entry
- **Personalise AI agent tone** — Low Risk gets empathetic nudges, High Risk gets structured intervention
- **Prioritise escalation** — High Risk borrowers with large outstanding balances flagged for human review
- **Multilingual extension** — scoring pipeline can feed into vernacular AI conversation engines
 
---
 
## 📈 Future Work
 
- Apply framework to real NPA (Non-Performing Asset) data from Indian NBFCs
- Add SHAP explainability for per-borrower risk explanations
- Build a real-time scoring API (FastAPI) for integration into collections platforms
- Extend to multilingual borrower profiling using regional language NLP
 
---
  
## 👤 Author
 
**Ashmal Abdussalam P T**   
