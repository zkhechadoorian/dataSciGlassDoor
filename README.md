# 🧠 Data Science Salary Estimator

A complete end-to-end machine learning project that predicts data science salaries based on job descriptions. It includes web scraping, data cleaning, feature engineering, model tuning, and API deployment using Flask.

---

## 📊 Project Overview

- Created a tool that estimates data science salaries (MAE ~ $11K) to help professionals negotiate better offers.
- Scraped over 1,000 job descriptions from Glassdoor using Python and Selenium.
- Engineered features from job descriptions to quantify the importance of Python, Excel, AWS, and Spark.
- Trained and tuned Linear, Lasso, and Random Forest Regressors using GridSearchCV.
- Deployed a production-ready Flask API for real-time predictions.

---

## 🗃️ Code & Resources Used

- **Python Version**: 3.10+
- **Main Packages**: `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `flask`, `selenium`, `pickle`
- **Setup Requirements**:  
    ```bash
    pip install -r requirements.txt
    ```

---

## 🧹 Web Scraping

Using a customized scraper, we collected:

- Job Title  
- Salary Estimate  
- Description  
- Company Info  
- Headquarters, Size, Age  
- Industry, Sector, Revenue  
- Competitors  

Results are stored in 
---

## 🧼 Data Cleaning

Created custom features:

- Parsed salaries (hourly, employer-provided)  
- Extracted company rating, state, and age  
- Flagged skills: Python, R, Excel, AWS, Spark  
- Simplified job title & seniority  
- Computed description length  

---

## 📈 Exploratory Data Analysis (EDA)

Used pivot tables and visualizations to explore:

- Salary by job title  
- Job opportunities by state  
- Correlations among features  

---

## 🧠 Model Building

- Converted categorical variables to dummy variables  
- Split data (80% train, 20% test)  
- Evaluated using MAE (Mean Absolute Error)  

### Models Tested:
- Multiple Linear Regression (baseline)  
- Lasso Regression (handles sparsity)  
- Random Forest (best performance)  

---

## 📊 Model Performance

| Model              | MAE   |
|--------------------|-------|
| Random Forest      | 11.22 |
| Linear Regression  | 18.86 |
| Ridge Regression   | 19.67 |

---

## 🚀 Productionization (Flask API)

### API Workflow:
1. Takes job data as JSON input  
2. Returns predicted salary  

---

## 🛠️ How to Run This Project (Windows / Linux / macOS)
```
cd path/to/repository/directory/  
```

2️⃣ **Create a Virtual Environment**  

For macOS/Linux:  
```bash
python3 -m venv .venv # (it creates a virtual environment named .venv)
source .venv/bin/activate # (it activates the virtual environment)
```

For Windows:  
```bash
python -m venv .venv (it creates a virtual environment named .venv)
.venv\Scripts\activate (it activates the virtual environment)
```

3️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt # (it installs all the dependencies)
```
4️⃣ **Load and  Clean Data**

Next run the data_cleaning.py file using this command:
```bash
python data_cleaning.py
```

Perform EDA and prepare the input for model building in the next step:
```
1. Open `data_cleaning.ipynb`
2. select the kernel .venv/bin/activate
3. run all cells. (it loads the data and cleans it)
```


4️⃣ **Train and Save Models**  
```bash
python model_building.py #(it trains the model and saves it in Flask/model/model_file.p)
```

5️⃣ **Start the Flask API**  
```bash
python FlaskAPI/app.py #(it starts the Flask API server)
```

Server runs at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

6️⃣ **Test the API**

- **Option 1**: Python script  
    ```bash
    python FlaskAPI/make_request.py ##(it makes a request to the API)
    ```

- **Option 2**: curl (replace with valid path)  
    ```bash
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @FlaskAPI/data_input.json #(it makes a request to the API using curl)
    ```

- **Option 3**: Postman  
    - Method: POST  
    - URL: [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)   (it makes a request to the API using Postman)
    - Body: raw → JSON → paste from `sample_input.json`  

---

## 🧾 Directory Structure

```plaintext
Predict_DataScience_Salary/
├── model_building.py
├── eda_data.csv
├── requirements.txt
├── random_forest_model.pkl
├── FlaskAPI/
│   ├── app.py
│   ├── make_request.py
│   └── sample_input.json
```

---

## 📌 Final Note

The trained Random Forest model is used in production via Flask and achieves a MAE of ~11K. This tool is designed to empower data scientists to better understand the market value of their skills.
