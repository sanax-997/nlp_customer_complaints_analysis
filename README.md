# Topic Modelling - NLP Analysis of Consumer Complaints

## Overview
The "NLP Analysis of Consumer Complaints" repository contains the code and data for the topic modeling of a telecom customer complaints dataset. The primary objective of this project is to test two different topic modeling approaches on the dataset and compare their respective results. This repository contains all the steps from the preprocessing of data to the visualization of the topic modeling.

## Installation
1. Download the application from Github and extract the contents
2. If you do not have `pipenv` already installed. Open a code command prompt or terminal in the extracted folder and install `pipenv` with the command `pip install pipenv`
3. Then execute the command `pipenv install`. This installs all the dependencies needed for the program
4. Execute the command `pipenv shell` to open a new virtual environment

## Usage
To use the application follow the steps:
1. Change the variable `nlp_method` to either `tfidf_lda` to apply tf-idf and lda or `Bertopic` to apply the BERTopic pipeline
2. In the terminal execute the command `py main.py`
   
## Requirements
Python Version 3.9.0

## License
GPL-3.0 License
