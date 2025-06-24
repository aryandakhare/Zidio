Data Science & Analytics Intern - Zidio Development
Duration: May 2025 - July 2025

This repository contains the projects and analyses conducted during my Data Science & Analytics Internship at Zidio Development. The primary focus was on stock market analysis and forecasting using various machine learning and deep learning models.

Table of Contents
Project Overview

Key Achievements & Responsibilities

Technologies & Libraries Used

Project Structure

Future Enhancements

Contact

Project Overview
During this internship, I developed a comprehensive stock market analysis and forecasting system. The project involved:

Data Acquisition and Preprocessing: Sourcing historical stock data and preparing it for model training, including handling missing values and scaling.

Exploratory Data Analysis (EDA): Visualizing historical stock prices, volume, and correlations to understand market trends and relationships between different financial indicators.

Model Development: Implementing and evaluating several time series forecasting models, including:

Long Short-Term Memory (LSTM): A deep learning model well-suited for sequence prediction tasks.

AutoRegressive Integrated Moving Average (ARIMA): A classical statistical method for time series forecasting.

Seasonal AutoRegressive Integrated Moving Average (SARIMAX): An extension of ARIMA that also handles seasonality and exogenous variables.

Prophet: A forecasting model developed by Facebook, known for its ability to handle seasonality, holidays, and trends.

Regression Models (Linear Regression, SVR, Decision Tree): Applied for baseline comparisons or feature-based predictions.

Model Evaluation: Assessing model performance using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

Deployment (Streamlit): Developing an interactive web application using Streamlit to visualize data, forecasts, and allow users to interact with the models.

Key Achievements & Responsibilities
Successfully implemented and fine-tuned multiple forecasting models, demonstrating proficiency in both traditional time series and deep learning approaches.

Conducted thorough data cleaning and exploratory data analysis to derive meaningful insights from financial data.

Developed a user-friendly Streamlit application, enhancing the accessibility and interpretability of the forecasting results.

Contributed to the data science team by providing actionable insights and robust predictive models for stock market trends.

Gained practical experience in end-to-end data science project lifecycle, from problem definition to deployment.

Technologies & Libraries Used
Programming Language: Python

Data Manipulation: pandas, numpy

Data Visualization: matplotlib, seaborn, plotly

Stock Data Acquisition: yfinance

Machine Learning: scikit-learn (for MinMaxScaler, KMeans, PCA, SVR, DecisionTreeRegressor, LinearRegression, mean_squared_error, mean_absolute_error)

Time Series Analysis: statsmodels (for ARIMA, SARIMAX), prophet

Deep Learning: tensorflow, keras (for Sequential, Dense, LSTM, EarlyStopping)

Web Application Framework: streamlit

Tunneling (for local deployment): pyngrok

Project Structure
Stock_Market_analysis_and_forecast_.ipynb: The primary Jupyter Notebook containing the detailed analysis, model training, and evaluation for stock market forecasting. This notebook likely covers the LSTM, ARIMA, SARIMAX, and Prophet models.

zidio.ipynb: Another Jupyter Notebook, possibly an earlier iteration or a separate exploration, which also includes stock data loading, cleaning, and LSTM model building. It appears to focus on Tesla stock data.

streamlit_app.py: (Implied from the Stock_Market_analysis_and_forecast_.ipynb output) The Python script for the Streamlit web application that deploys the stock forecasting models.

How to Run (if applicable)
To replicate the analysis or run the Streamlit application:

Clone the repository:

git clone <repository_url_here>
cd <repository_name_here>

Install dependencies:
It is recommended to use a virtual environment.

pip install pandas matplotlib seaborn plotly yfinance scikit-learn statsmodels tensorflow keras prophet streamlit pyngrok

(Note: Ensure your numpy version is compatible with tensorflow and prophet. If you encounter issues, try installing numpy>=1.26 first or follow specific version recommendations from the library documentation.)

Run the Jupyter Notebooks:
You can open and run Stock_Market_analysis_and_forecast_.ipynb or zidio.ipynb using Jupyter Lab or Jupyter Notebook.

jupyter notebook

or

jupyter lab

Run the Streamlit application:
If streamlit_app.py is available, you can run it as follows:

streamlit run streamlit_app.py

If the notebook uses pyngrok for public URL exposure, ensure you have an ngrok authentication token configured.

Future Enhancements
Integration of more advanced financial indicators and alternative data sources.

Development of a real-time data ingestion pipeline.

Implementation of more sophisticated deep learning architectures (e.g., Transformers).

Ensemble modeling to combine predictions from multiple models for improved accuracy.

Adding interactive features for users to select different stocks and date ranges.

Contact
Aryan Gajanan Dakhare
dakharearyan863@gmail.com
Linkkdin: https://www.linkedin.com/in/aryandakhare?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
Git Hub: https://github.com/aryandakhare
