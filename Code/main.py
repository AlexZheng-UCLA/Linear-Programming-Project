from MyRegressor import MyRegressor
from utils import prepare_data_gaussian, prepare_data_news, plot_result
import numpy as np

data = prepare_data_gaussian() 
# data = prepare_data_news()
trainX = data['trainX']
trainY = data['trainY']
N_train = trainX.shape[0]

testX = data['testX']
testY = data['testY']
N_test = testX.shape[0]

train_error = []
test_error = []

np.random.seed(0)
opt_alpha = 1e-2

# TASK 1-2 **************************************
alpha_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
# alpha_list = [6e-3, 8e-3, 1e-2, 1.2e-2, 1.4e-2, 1.6e-2, 1.8e-2, 2e-2]
for alpha in alpha_list:
    regressor = MyRegressor(alpha=alpha)
    train_err = regressor.train(trainX, trainY)
    train_error.append(train_err)

    _, test_err = regressor.evaluate(testX, testY)
    test_error.append(test_err)

plot_result({'taskID':'1-2', 'alpha':alpha_list, 'train_err':train_error, 'test_err':test_error})

print(f"train_error: {train_error}")
print(f"test_error: {test_error}")

## TASK 1-3 ***********************************************************************

# old_regressor = MyRegressor(alpha=opt_alpha)
# old_regressor.train(trainX, trainY)

# feat = old_regressor.select_features()
# regressor = MyRegressor(alpha=0)
# feat_num = []
# train_error = []
# test_error = []

# for per in [0.01, 0.1, 0.3, 0.5, 1]:

#     feat_num.append(per)
#     num = int(len(feat)*per)
#     selected_feat = feat[:num]

#     trainX_s = trainX[:, selected_feat]
#     testX_s = testX[:, selected_feat]
#     error = regressor.train(trainX_s, trainY)
#     train_error.append(error)

#     _, error = regressor.evaluate(testX_s, testY)
#     test_error.append(error)

# print(f"feature num: {feat_num}")
# print(f"train_error: {train_error}")
# print(f"test_error: {test_error}")
# plot_result({'taskID':'1-3', 'feat_num':feat_num, 'train_err':train_error, 'test_err':test_error})

## TASK 1-4 ***************************************************************

# old_regressor = MyRegressor(alpha=opt_alpha)
# old_regressor.train(trainX, trainY)
# selected_trainX, selected_trainY = old_regressor.select_sample(trainX, trainY)

# regressor = MyRegressor(alpha=0)
# sample_num = []
# train_error = []
# test_error = []

# for per in [0.01, 0.1, 0.3, 0.5, 1]:

#     sample_num.append(per)
#     num = int(N_train * per)

#     trainX_s = selected_trainX[:num]
#     trainY_s = selected_trainY[:num]

#     error = regressor.train(trainX_s, trainY_s)
#     train_error.append(error)

#     _, error = regressor.evaluate(testX, testY)
#     test_error.append(error)

# print(f"sample num: {sample_num}")
# print(f"train_error: {train_error}")
# print(f"test_error: {test_error}")
# plot_result({'taskID':'1-4', 'sample_num':sample_num, 'train_err':train_error, 'test_err':test_error})

## TASK 1-5 ***************************************************************

# old_regressor = MyRegressor(alpha=opt_alpha)
# old_regressor.train(trainX, trainY)

# regressor = MyRegressor(alpha=0)
# train_error = []
# test_error = []
# communication_cost = [0.1, 0.3]

# for per in communication_cost:

#     old_regressor.training_cost = per 
#     trainX_s, trainY_s = old_regressor.select_data(trainX, trainY)
#     selected_feat = old_regressor.features

#     error = regressor.train(trainX_s, trainY_s)
#     train_error.append(error)

#     testX_s = testX[:, selected_feat]
#     _, error = regressor.evaluate(testX, testY)
#     test_error.append(error)

# print(f"cost: {communication_cost}")
# print(f"train_error: {train_error}")
# print(f"test_error: {test_error}")
# plot_result({'taskID':'1-5', 'cost':communication_cost, 'train_err':train_error, 'test_err':test_error})





