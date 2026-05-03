#  Telco Customer Churn Analysis & Power BI Dashboard

A comprehensive end-to-end analytics project focused on customer churn behavior using Python for data processing and Power BI for interactive data visualization.


##  Project Objective

To identify key patterns and risk factors contributing to customer churn at a telecom company. This solution empowers business stakeholders to take proactive retention actions.


##  Dataset

Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Final File Used: `cleaned_telco_churn.csv` (preprocessed with Python)



## Tools & Technologies

| Category            | Stack                                 |
|---------------------|----------------------------------------|
| Data Cleaning       | Python (Pandas, NumPy)                |
| EDA & Visualization | Seaborn, Matplotlib                   |
| BI Dashboard        | Power BI Desktop                      |
| File Format         | CSV, PBIX                             |



##  Power BI Dashboard Highlights

## Key Insights Visualized
- Churn Rate by:
  - Gender, Contract Type, Internet Service, Payment Method
- Monthly Churn Trend
- KPI Cards:
  - Total Customers
  - Churned Customers
  - Churn % by Contract & Tenure


## Tooltips Enriched With
- Monthly Charges
- Total Charges
- Tech Support Status
- Contract Type

###  Drill-Down Filters
- Gender
- Senior Citizen Status
- Internet Service Type
- Payment Method
   


##  Python Workflow Summary

- Data cleaning & transformation
- Missing value imputation
- Label & One-Hot Encoding
- Outlier detection
- Exploratory data analysis (EDA)
- Exported cleaned dataset for BI use



##  Repository Structure

| Folder/File                             | Description                     |
|----------------------------------------|---------------------------------|
| `dataset.csv`                          | Original dataset               |
| `main.py`                              | Python script (run with `python main.py`) |
| `output/`                              | Generated charts (PNG files)   |
| `Telco Customer Churn Insights Dashboard.pbix` | Power BI dashboard file |
| `README.md`                            | Project documentation          |



##  How to Run

##  Key Results

| Metric | Value |
|--------|-------|
| Total Customers | 7,043 |
| Churned Customers | 1,869 (26.5%) |
| Retained Customers | 5,174 (73.5%) |
| Model Accuracy | 80.48% |
| Model ROC AUC | 84.21% |

### Top Churn Risk Factors
1. **Contract Type** - Month-to-month contracts have highest churn risk
2. **Tenure** - New customers (shorter tenure) are more likely to churn
3. **Internet Service** - Fiber optic customers show higher churn

---

### Python Analysis (main.py)

```bash
# Install dependencies
pip install pandas seaborn matplotlib scikit-learn xgboost

# Run the project
python main.py
```

This will:
- Load and clean the dataset
- Perform EDA and generate charts
- Train a churn prediction model
- Display key results in the terminal
- Save charts to the `output/` folder

### Power BI Dashboard

Open `Telco Customer Churn Insights Dashboard.pbix` in Power BI Desktop to view interactive visualizations.


## Author 
   About the Author
Harshita Sharma
🎓 Btech Student | Data Analytics Enthusiast
💼 Python | Power BI | SQL | Data Cleaning & EDA


##  Show Your Support
If you found this project useful or inspiring:

- Star this repo
- Fork and experiment
- Connect with me on LinkedIn


  
📌 License
This project is for educational and portfolio purposes only.