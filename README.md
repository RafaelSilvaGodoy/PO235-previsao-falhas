# RUL Prediction for Predictive Maintenance

Predictive maintenance techniques are employed to assess the condition of equipment, allowing for the proactive planning of maintenance activities and the anticipation of potential failures before they occur. This approach helps ensure that maintenance is conducted at the optimal time, preventing unexpected breakdowns and extending the equipment's operational lifespan.

The objective of this project is to implement some models to predict the Remaining Useful Life (RUL) of equipment and evaluate the performance of each one.

## Members
- Acélio Luna
- Joniel Bastos
- Rafael Godoy

## Data
The dataset comprises multiple multivariate time series, each representing data from a distinct engine, resembling a fleet of similar engines. Initially, each engine operates normally but eventually develops a fault. In the training set, the fault progresses to system failure, whereas in the test set, the series concludes before failure occurs. The training data includes operational information from 100 engines, with run lengths ranging from 128 to 356 cycles. Similarly, the test data also covers 100 different engines, entirely separate from those in the training set. You can find the data [here](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).

### Requirements

To run this project, you will need to have installed on your machine:

- Python 3.8 or later version
- Python libraries listed in `requirements.txt`
