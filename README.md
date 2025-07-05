## Food Delivery Times Prediction

#### Objective

This project predicts the estimated delivery time (in minutes) for food orders based on various features such as distance, weather, traffic level, preparation time, courier experience, and more. It is an end-to-end Machine Learning project with a deployed web app.

This helps optimize delivery operations and improve customer satisfaction.

#### Dataset Source

Kaggle Link: https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction

### Render Deployment Link

Check out the live web app: (https://food-delivery-time-prediction-wzyj.onrender.com)

#### Screenshot

![homepage](https://github.com/user-attachments/assets/f1428813-76ad-4b71-9b85-cb567f385fd2)

#### Input Features

- Distance (km)
- Weather
- Traffic Level
- Time of Day
- Vehicle Type
- Preparation Time (min)
- Courier Experience (yrs)

#### Predicted Output

- Delivery Time (minutes)

#### Features

- Exploratory Data Analysis (EDA)
- Preprocessing Pipeline
- Model Evaluation & Comparison
- Flask Web Interface
- Render Deployment
  
#### ML Models Used

- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor 
- Gradient Boosting Regressor
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

### Project Architecture & Pipeline

1. Data Ingestion

- Reads the raw CSV file from the local notebook/data folder.

- Performs train-test split with an 80/20 ratio.

- Saves train.csv, test.csv, and data.csv in the artifacts/ directory.

2. Data Transformation

- Builds a preprocessing pipeline using ColumnTransformer:

    - Numerical Features: Imputed using median strategy, then scaled with StandardScaler.

    - Categorical Features: Imputed with the most frequent strategy, encoded via OneHotEncoder, and scaled.

Saves the preprocessing object (preprocessor.pkl) for future use.

3. Model Training

- Trains and evaluates various regressors including:

- Linear, Decision Tree, Random Forest, XGBoost, CatBoost, etc.

- The best model is saved as a serialized file (model.pkl).

4. Prediction Pipeline

- Loads both the preprocessor and trained model.

- Accepts new input via a form, transforms the data, and returns predicted delivery time.

5. Flask Web App

- Built a clean user interface using HTML and CSS.

- Allows users to input real-world features and get delivery time prediction instantly.

- Deployed using Render for public access.

6. Requirements File

- The requirements.txt file lists all the Python libraries and versions used in this project, ensuring the same environment can be replicated anywhere.

### Exploratory Data Analysis Notebook

Link: https://github.com/5warna/Food-Delivery-Prediction/blob/master/notebook/EDA%20TIME%20PREDICTION.ipynb

### Model Training Notebook

Link: https://github.com/5warna/Food-Delivery-Prediction/blob/master/notebook/Model%20Training.ipynb

****

#### Acknowledgements

- Special thanks to Krish Naik for his valuable tutorials and guidance on end-to-end machine learning project deployment.
- His content helped shape the structure, modularization, and deployment flow of this project.



