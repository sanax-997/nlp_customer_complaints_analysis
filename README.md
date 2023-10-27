# Topic Modelling - NLP Analysis of Consumer Complaints

## Overview
The "Demand Forecasting Model for Public Transport" repository contains the code and data for the analysis of taxi demand in key areas of New York City and the development of a prediction model. The primary objective of this project is to accurately predict demand patterns based on historical taxi pickup data in the ten most important locations of the city. This repository contains all the steps from the preprocessing of data to the evaluation of the finished prediction model. The .ipynb scripts in the repository contain the following:
- feature_generation.ipynb - this script contains all code regarding feature preprocessing
- eda.ipynb (exploratory data analysis - is about the visual exploration of the data
- model_training.ipynb . is about the model training and prediction

## Installation
1. Download the application from Github and extract the contents
2. Open the folder with an IDE able to execute .ipynb files
3. Download the Uber Pickups in New York City dataset from [Kaggle](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city)
4. Select only the Uber trip data from 2014 (April - September)
5. Rename the files to "data-(month)(year).csv"
6. Move the files to the "Raw Data" subfolder

## Usage
To use the application follow the steps:
1. Select the feature_generation.ipynb file
2. Execute the "Run All" command to generate the .csv files for eda and model training
3. Select the eda.ipynb file
4. Execute the "Run All" command to generate a graphical representation of the dataset
5. Select the model_training.ipynb file
6. Execute the "Run All" command to train regression models and make predictions
   
## Requirements
Python Version 3.9.0

## License
GPL-3.0 License
