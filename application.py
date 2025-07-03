from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Home route
# @app.route('/')
# def index():
    # return render_template('index.html')  # Optional landing page

# Prediction form route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Shows form
    
    else:
        try:
            # Fetching values from form
            data = CustomData(
                Distance_km=float(request.form.get('Distance_km')),
                Weather=request.form.get('Weather'),
                Traffic_Level=request.form.get('Traffic_Level'),
                Time_of_Day=request.form.get('Time_of_Day'),
                Vehicle_Type=request.form.get('Vehicle_Type'),
                Preparation_Time_min=int(request.form.get('Preparation_Time_min')),
                Courier_Experience_yrs=float(request.form.get('Courier_Experience_yrs')),
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_dataframe()
            print("Input Data:\n", pred_df)

            # Load pipeline and predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=f"{results[0]:.2f} minutes")

        except Exception as e:
            print("Prediction Error:", e)
            return render_template('home.html', results="Something went wrong ❌")
        

if __name__ == "__main__":
    app.run(host="0.0.0.0") 