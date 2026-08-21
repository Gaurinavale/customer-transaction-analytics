# 🛒 Customer Transaction Analytics

An end-to-end **Data Analytics project** focused on cleaning, analyzing, and understanding customer transaction data.

## 📌 What I Did

* Cleaned and preprocessed raw customer transaction data
* Handled missing values and duplicates
* Performed Exploratory Data Analysis (EDA)
* Analyzed customer and transaction behavior
* Created new features using Feature Engineering
* Identified business insights and patterns
* Built an interactive **Streamlit dashboard**

## 📊 Dataset Columns

| Column                   | Description                 |
| ------------------------ | --------------------------- |
| `transaction_id`         | Unique transaction ID       |
| `customer_id`            | Unique customer ID          |
| `transaction_date`       | Date of transaction         |
| `transaction_amount`     | Total transaction amount    |
| `item_price`             | Price of the item           |
| `quantity`               | Number of items purchased   |
| `product_category`       | Product category            |
| `product_name`           | Product name                |
| `payment_method`         | Payment method used         |
| `customer_dob`           | Customer date of birth      |
| `account_creation_date`  | Account creation date       |
| `last_login_date`        | Customer's last login date  |
| `loyalty_program_member` | Loyalty program membership  |
| `device_used`            | Device used for transaction |
| `shipping_address_state` | Customer's state            |
| `transaction_status`     | Transaction status          |

## 🔄 Project Architecture

```text
                Raw Customer Data
                       │
                       ▼
               Data Cleaning
                       │
                       ▼
              Data Preprocessing
                       │
                       ▼
                  EDA Analysis
                       │
                       ▼
             Feature Engineering
                       │
                       ▼
              Business Insights
                       │
                       ▼
            Streamlit Dashboard
```

### Workflow

**Raw Data → Cleaning → EDA → Feature Engineering → Business Insights → Dashboard**

## 📊 Analysis

The project includes:

* Customer spending analysis
* Product and category analysis
* Transaction trends
* Payment method analysis
* Customer age analysis
* Loyalty program analysis
* Outlier detection
* Correlation analysis

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Streamlit
* Jupyter Notebook
* Git & GitHub

## 📁 Project Structure

```text
customer-transaction-analytics/
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/
│   └── customer_transaction_analysis.ipynb
│
├── README.md
└── .gitignore
```

## ▶️ Run Locally

```bash
git clone https://github.com/Gaurinavale/customer-transaction-analytics.git
cd customer-transaction-analytics
```

Run the dashboard:

```bash
https://customer-transaction-analytics-k6mhyjhdrjftaj3oank34z.streamlit.app/
```

## 📓 Notebook

[Customer Transaction Analysis Notebook](notebooks/customer_transaction_analysis.ipynb)

## 🚀 Future Improvements

* Customer Segmentation
* Churn Prediction
* Customer Lifetime Value Prediction
* Sales Forecasting
* Recommendation System


---

⭐ If you find this project useful, feel free to star the repository!
