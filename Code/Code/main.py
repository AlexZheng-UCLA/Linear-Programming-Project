from utils import plot_result
from MyRegressor import MyRegressor
from utils import prepare_data_gaussian, prepare_data_news, preprocessing
import numpy as np
import numpy.linalg as la

## TASK 1 ***************************************
data = prepare_data_gaussian() 
trainX = data['trainX']
trainY = data['trainY']
N_train = trainX.shape[0]

testX = data['testX']
testY = data['testY']
N_test = testX.shape[0]

alpha = [1e-5, 1e-4, 1e-3, 1e-2, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 1.25e-2]
train_error = []
test_error = []

for a in alpha:
    regressor = MyRegressor(a)
    train_error.append(regressor.train(trainX, trainY))

    y_pred = np.matmul(testX, regressor.weight.reshape(-1, 1)) + np.ones((N_test, 1)) * regressor.bias
    test_error.append(la.norm(testY.reshape(-1, 1) - y_pred, ord=1) / N_test)

plot_result({'taskID':'1-2', 'alpha':alpha, 'train_err':train_error, 'test_err':test_error})

print(f"train_error: {train_error}")
print(f"test_error: {test_error}")



