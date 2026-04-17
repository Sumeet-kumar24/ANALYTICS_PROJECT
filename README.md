# 🛒 Online Retail Intelligence Dashboard

An interactive Streamlit dashboard for exploratory data analysis, outlier detection, hypothesis testing, and Customer Lifetime Value (CLV) modelling on the **UCI Online Retail dataset**.

---

## 📸 Features

| Section | What it does |
|---|---|
| **KPI Cards** | Total orders, revenue, average order value, unique customers |
| **Order Value Distribution** | Histogram + KDE (95th-pct capped) |
| **Monthly Revenue Trend** | Time-series line chart by calendar month |
| **Outlier Detection** | Log-transformed box plots for Quantity, UnitPrice, Revenue |
| **Correlation Heatmap** | Pearson correlation matrix across numeric features |
| **Top 10 Products** | Horizontal bar chart ranked by total revenue |
| **Hypothesis Testing** | Welch's T-Test: UK vs International average order value |
| **CLV OLS Regression** | Frequency → Total Spend regression with R² score |
| **CLV Predictor Tool** | Interactive slider to estimate spend by purchase frequency |

All panels respond to the **Country filter** in the sidebar.

---

## 🗂 Project Structure

```
retail_dashboard/
│
├── app.py                  # Main Streamlit application
├── online_retail.csv       # Dataset (UCI Online Retail)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/retail-dashboard.git
cd retail-dashboard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📦 Dataset

**UCI Online Retail Dataset**  
- ~541 K transaction records across 38 countries (2010–2011)  
- Columns: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`  
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)

### Preprocessing steps applied in `app.py`

1. Drop rows with missing `CustomerID` or `InvoiceNo`  
2. Remove cancelled orders (InvoiceNo prefixed with `C`)  
3. Filter out non-positive `Quantity` and `UnitPrice`  
4. Engineer `Revenue = Quantity × UnitPrice`  
5. Parse `InvoiceDate` and extract `Month`

---

## 🧪 Statistical Methods

- **Welch's T-Test** — compares means of two independent samples with unequal variances; no `scipy` dependency (computed manually with NumPy)  
- **OLS Regression** — `numpy.polyfit` for degree-1 polynomial fit; R² computed from residual / total sum of squares  

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework |
| [Pandas](https://pandas.pydata.org) | Data wrangling |
| [NumPy](https://numpy.org) | Numerical computing |
| [Matplotlib](https://matplotlib.org) | Base plotting |
| [Seaborn](https://seaborn.pydata.org) | Statistical visualisation |

---

## 📄 License

MIT License — free to use, modify, and distribute.
