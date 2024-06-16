# RUL Prediction for Predictive Maintenance

Predictive maintenance techniques are employed to assess the condition of equipment, allowing for the proactive planning of maintenance activities and the anticipation of potential failures before they occur. This approach helps ensure that maintenance is conducted at the optimal time, preventing unexpected breakdowns and extending the equipment's operational lifespan.

The objective of this project is to implement some models to predict the Remaining Useful Life (RUL) of equipment and evaluate the performance of each one.

## Members
- Acélio Luna
- Joniel Bastos
- Rafael Godoy

## Data
The dataset comprises multiple multivariate time series, each representing data from a distinct engine, resembling a fleet of similar engines. Initially, each engine operates normally but eventually develops a fault. In the training set, the fault progresses to system failure, whereas in the test set, the series concludes before failure occurs. The training data includes operational information from 100 engines, with run lengths ranging from 128 to 356 cycles. Similarly, the test data also covers 100 different engines, entirely separate from those in the training set. You can find the data [here](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).

The exploratory data analysis can be found in [scripts/eda.ipynb](https://github.com/RafaelSilvaGodoy/PO235-previsao-falhas/blob/befa0470a9ce7e4372e8ef83c6f31e6e259f6ccd/scripts/eda.ipynb)

The `train_set.csv` and `test_set.csv` are expected to be a csv file with following structure

![Dataset Columns](https://github.com/RafaelSilvaGodoy/PO235-previsao-falhas/blob/5c81e24577febf5a0820c299c71244374b3b3a95/dataset/images/table.png)

### split_folders

Used for model search. Result of the 10-fold 10 times with different seeds in the dataset. In case the dataset changes, the split_dataset code can be run to recreate news splits.

## Models for RUL Prediction
(1) Exponential Degradation

(2) LSTM model

(3) XGBoost model

All codes are documented for reproduction.

## Model Search
The model search is performed with 10 repetitions of the 10-fold with different seeds (the code to generate this can be found in [scripts/split_dataset.ipynb](https://github.com/RafaelSilvaGodoy/PO235-previsao-falhas/blob/cea7f798dd38e052d2a0b529b9de7541a2d475b3/scripts/split_dataset.ipynb)). For each model pipeline, run `python scripts/{model_name}.py`. The results are saved in **models** folder.

To validation, run `python scripts/report.py`. The report is saved in `Validation_Report.pdf`

## Project Production

## Requirements

To run this project, you will need to have installed on your machine:

- Python 3.8 or later version
- Python libraries listed in `requirements.txt`
