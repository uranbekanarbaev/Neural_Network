# Property Price Prediction
Overview
This project aims to predict the sale prices of residential properties using both linear regression and neural network algorithms. The analysis includes exploratory data analysis, model development, and evaluation.

Data Analysis
The dataset consists of various features such as MSSubClass, LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt, and others, along with the target variable SalePrice. Data preprocessing involved handling missing values and scaling numerical features. Exploratory data analysis revealed correlations between certain features and the target variable.

Visualization
One plot was created to visualize the distribution of the SalePrice variable. The plot highlighted the presence of outliers, which were addressed during preprocessing to improve model performance.

Linear Regression Algorithm
The linear regression model was trained and evaluated using scikit-learn. Evaluation metrics including Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) were computed on the test dataset. The model achieved impressive performance with MAE: 9.42e-11, MSE: 1.95e-20, and RMSE: 1.40e-10.

Neural Network
A neural network model was developed using TensorFlow/Keras. The architecture included multiple dense layers with activation functions. The model was trained on the same dataset and evaluated using the same metrics. The neural network demonstrated competitive performance compared to linear regression, providing insights into the predictive power of deep learning techniques for property price prediction.

Conclusion
Both linear regression and neural network models showcased strong predictive capabilities for property price prediction. The evaluation metrics indicated minimal errors, suggesting that the models effectively captured the underlying patterns in the data. Further experimentation and model refinement could potentially enhance performance even further.

Requirements
Python 3.x
scikit-learn
TensorFlow
pandas
numpy
matplotlib
Usage
To run the code:

Clone this repository.
Install the required dependencies.
Execute the notebooks or scripts provided.
Author
ANARBAEV URANBEK

Contact: uranbekanarbaev@gmail.com

License
This project is licensed under the [License Name]. See the LICENSE file for details.
