# Forecast of Colombian Stock Exchange indices

This code is a Jupyter notebook that was created on Google Colab. This notebook is a study case for predicting trends in stock market indices using machine learning models for the Colombian stock exchange market, specifically the COLCAP stock market index.

## Code Description
This code is designed to compare the accuracy of two models (ANN and SVR) for predicting the COLCAP index. To determine the most accurate model, a deviation matrix is calculated to find the model with the smallest mean deviation.

The code produces a table of results for the applied models and a graphical comparison of the real and predicted values for each model. It also calculates the mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE) for each model and presents them in a comparison matrix. Additionally, a matrix of percentage deviations is calculated, and the mean deviation for each algorithm is presented to select the best model.

## Instructions
To run this code, you need to have the following packages installed:

Pandas
Matplotlib
Seaborn
Scikit-learn
After installing the required packages, copy and paste the code into your Python environment and run it. The results will be displayed in tables and graphs.

##Outputs
The code produces the following outputs:

Table of results for the applied models
Graphical comparison of the real and predicted values for each model
Comparison matrix of the mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE) for each model
Matrix of percentage deviations and mean deviation for each algorithm to select the best model

## Libraries
The required libraries for data processing and model implementation are imported in this code.
* pandas
* numpy
* matplotlib.pyplot
* seaborn
* math
* sklearn.preprocessing
* keras.models
* keras.layers
* sklearn.model_selection
* sklearn.svm
* keras.callbacks
* sklearn.neural_network
* sklearn.metrics
* datetime

## Data Selection
The information is imported from the original file Indices Consolidado.xlsx. A dataframe is constructed with the information of the analyzed stock market index. The data is filtered for only the COLCAP index, and irrelevant columns are dropped. The information is then summarized, and the time series plot is shown.

## Data Preparation
The data is prepared as a time series, where the input vector is constructed with the previous 6 days to predict the day 7. The dataset is normalized with the Min-Max Scaler.

## Data Splitting
The data is split into training and testing datasets. The training dataset comprises 75% of the original dataset, and the testing dataset contains the remaining 25% of the dataset. Finally, the training and validation datasets are reshaped to match the input shape required by the models.

## Modeling Technique Selection
The problem is modeled as a time series where the value of the index varies over time. That is, it corresponds to a supervised regression learning problem. The input vectors correspond to the value of the index in the previous 6 days.

## Algorithm Selection 1 - ANN
The first model corresponds to an artificial neural network of the multilayer perceptron type.

## Define Hyperparameters and Create Model
For this multilayer perceptron, a neural network is defined as follows:
- An input layer consisting of 6 neurons given the 6 previous days of reference.
- A hidden layer with 6 neurons and a hyperbolic tangent activation function.

The code block crear_modeloFF() defines and compiles the model.

## Algorithm Training
The neural network is trained with the defined parameters and for 100 epochs.

## Prediction with Test Data
Initial results of the training are presented in a scatter plot and a loss graph.

## Cross-validation
KFold cross-validation technique is implemented with 10 splits.

## Prediction with Test Data
The model is used to make predictions with the test dataset. The results are normalized and a DataFrame is created with the real value, predicted value, and the difference between them.

Finally, the results are presented graphically.

## Algorithm 2 - SVR Selection
The second algorithm corresponds to Support Vector Machine Regression or better known as SVR.

The code first divides the data into training and testing datasets. The training dataset has 75% of the data, while the testing dataset has 25%. Then, the code reshapes the training and testing datasets for the model.

The kernel function used for this algorithm is radial basis function (RBF).

The code then trains the SVR model with the parameters created for 100 epochs. After that, it uses the trained model to predict the testing dataset.

The code then performs cross-validation using the KFold technique with 10 splits. For each split, the model is trained and tested, and the scores are printed out.

Finally, the code uses the trained model to predict the testing dataset. It then descales the predicted results and constructs a dataframe with the real value, the predicted value, and the difference between them.
