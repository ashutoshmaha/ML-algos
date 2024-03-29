# -*- coding: utf-8 -*-
"""Darts

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GFJ8VtzYSxcm0cu2IY47WQasr8T6-3kI
"""

#Darts for Time series analysis and Forecast using NBEATS Model

AirPassengersDataset().load().head(4)

MonthlyMilkDataset().load().head(4)

import matplotlib.pyplot as plt

dataset1 = AirPassengersDataset().load() #optionally, both can use converted to pd dataframe
dataset2 = MonthlyMilkDataset().load()

#plotting existing data 

dataset1.plot(label='Number of passengers') 
dataset2.plot(label='Pounds of milk per one cow')
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (20,4)
plt.legend(loc='lower right')
plt.title("Original Data")

#scaling the data and fitting

from darts.dataprocessing.transformers import Scaler

dataset1_scaling, datase2_scaling = Scaler(), Scaler()
dataset1_inScale = dataset1_scaling.fit_transform(dataset1)
dataset2_inScale = datase2_scaling.fit_transform(dataset2)

dataset1_inScale.plot(label='Passanger Data')
dataset2_inScale.plot(label='Milk Generation')
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (20,4)
plt.legend(loc='lower right')
plt.title("Scaled Data")

# Train test Splitting
dataset1_Training, val_passanger = dataset1_inScale[:-28], dataset1_inScale[-28:]
dataset2_Training, val_milkGeneration = dataset2_inScale[:-28], dataset2_inScale[-28:]

from darts import TimeSeries

from darts.utils.timeseries_generation import gaussian_timeseries, linear_timeseries, sine_timeseries

from darts.models import NBEATSModel

from darts.metrics import mape, smape

#Using NBEATS model for training the model

Gen_model = NBEATSModel(input_chunk_length=24, output_chunk_length=12, n_epochs=100, random_state=0)  #first 24 months as training data in every epoch
Gen_model.fit([dataset1_Training, dataset2_Training], verbose=True)

#Forecast for air passanger data

forecasting = Gen_model.predict(n=28, series=dataset1_Training)

dataset1_inScale.plot(label='Original Data')
forecasting.plot(label='Forecasted Values')

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (20,4)
plt.legend(loc='lower right')
plt.title("Air Passangers Forecast")


print('MAPE = {:.2f}%'.format(mape(dataset1_inScale, forecasting)))

#Forecast for milk generation


forecasting2 = Gen_model.predict(n=28, series=dataset2_Training)

dataset2_inScale.plot(label='Original Data')
forecasting2.plot(label='Forecasted Data')

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (20,4)
plt.legend(loc='lower right')
plt.title("Milk Generation Forecast")

print('MAPE = {:.2f}%'.format(mape(dataset2_inScale, forecasting2)))

#Ashutosh Mahajan