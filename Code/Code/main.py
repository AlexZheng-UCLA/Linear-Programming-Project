from utils import plot_result
from MyRegressor import MyRegressor
from utils import prepare_data_gaussian, prepare_data_news, preprocessing
import numpy as np
import numpy.linalg as la
from sklearn.metrics import mean_absolute_error


data = prepare_data_gaussian() 
trainX = data['trainX']
trainY = data['trainY']
N_train = trainX.shape[0]

testX = data['testX']
testY = data['testY']
N_test = testX.shape[0]

train_error = []
test_error = []

np.random.seed(0)

# TASK 1-2 **************************************
alpha = [0, 1e-5, 1e-4, 1e-3, 1e-2, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 1.25e-2, 1.5e-2]
# alpha = [6e-3, 8e-3, 1e-2, 1.2e-2, 1.4e-2, 1.6e-2, 1.8e-2, 2e-2]
for a in alpha:
    regressor = MyRegressor(a)
    train_error.append(regressor.train(trainX, trainY))

    y_pred = np.matmul(testX, regressor.weight.reshape(-1, 1)) + np.ones((N_test, 1)) * regressor.bias
    y_pred = y_pred.squeeze(1)
    test_error.append(mean_absolute_error(testY, y_pred))
    # test_error.append(la.norm(testY.reshape(-1, 1) - y_pred, ord=1) / N_test)

plot_result({'taskID':'1-2', 'alpha':alpha, 'train_err':train_error, 'test_err':test_error})

print(f"train_error: {train_error}")
print(f"test_error: {test_error}")

## TASK 1-3 **************************************
# old_regressor = MyRegressor(alpha=1.6e-2)
# old_regressor.train(trainX, trainY)
# selected_feat = old_regressor.select_features()
# print("selected_feat: ", selected_feat)

# regressor = MyRegressor(alpha=0)
# trainX_s = trainX[:, selected_feat]
# trainY_s = trainY[:, selected_feat]
# regressor.train(trainX_s, trainY_s)








