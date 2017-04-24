# Algorithm for DCF Analysis developed by Rohan Raval and Amar Singh
import scraper
import numpy as np
import pandas as pd
from scipy import sparse
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


#### MAIN METHOD #####

input_var = raw_input("Enter variable: ")
# get a dictionary of x:y pairs from dataset
# x = world-ranks
# y = input parameter, e.g. research score (out of 100)
data = scraper.getData(input_var)

df = pd.DataFrame.from_dict(data.items()) # convert data to a dataframe
x = df[0].to_frame() # x-values from the dataframe
y = df[1].to_frame() # y-values from the dataframe


### linear modeling ###

# make the models
ols_reg = linear_model.LinearRegression() #ordinary least squares
ridge_reg = linear_model.Ridge() #ridge regression
lasso_reg = linear_model.Lasso() #lasso regression
LARS_reg = linear_model.LassoLars() #least angle regression (on lasso)
b_ridge_reg = linear_model.BayesianRidge() #bayesian ridge regression
ard_reg = linear_model.ARDRegression() #bayesian ARD regression
sgd_reg =  linear_model.SGDRegressor() #stochastic gradient descent regression
ransac_model = linear_model.RANSACRegressor(ols_reg) #fit linear model with RANdom SAmple Consensus algorithm


# fit the models to a regression function based on our data points
ols_reg.fit(x,y)
ridge_reg.fit(x,y)
lasso_reg.fit(x,y)
LARS_reg.fit(x,y)
b_ridge_reg.fit(x,y)
ard_reg.fit(x,y)
sgd_reg.fit(x,y)
ransac_model.fit(x,y)

### validation of data (based on R-squared value) ### 

r_squared_scores= {
	'ols_scores' : ols_reg.score(x,y),
	'ridge_scores' : ridge_reg.score(x,y),
	'lasso_scores' : lasso_reg.score(x,y),
	'LARS_scores' : LARS_reg.score(x,y),
	'b_ridge_scores' : b_ridge_reg.score(x,y),
	'ard_scores' : ard_reg.score(x,y),
	'sgd_scores' : sgd_reg.score(x,y),
	'ransac_scores' : ransac_model.score(x,y)
}
vals = list(r_squared_scores.values())
keys = list(r_squared_scores.keys())

max_r_squared = keys[vals.index(max(vals))]
print "Model = %s" % max_r_squared
print "R-squared = %f" % r_squared_scores[max_r_squared] 

predicted = []
if max_r_squared == 'ols_scores':
	predicted = ols_reg.predict(x)
elif max_r_squared == 'ridge_scores':
	predicted = ridge_reg.predict(x)
elif max_r_squared == 'lasso_scores':
	predicted = lasso_reg.predict(x)
elif max_r_squared == 'LARS_scores':
	predicted = LARS_reg.predict(x)
elif max_r_squared == 'b_ridge_scores':
	predicted = b_ridge_reg.predict(x)
elif max_r_squared == 'ard_scores':
	predicted = ard_reg.predict(x)
elif max_r_squared == 'sgd_scores':
	predicted = sgd_reg.predict(x)
else:
	predicted = ransac_model.predict(x)

# algorithm
"""
reg = linear_model.LinearRegression() # do the linear regression
reg.fit(x, y) # get the linear fit, using x and y(revenues from df)
m = reg.coef_[0][0] # slope of fitted line
b = reg.intercept_[0] # intercept of fitted line

predicted = reg.predict(x)
"""
#print m, b

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.text(20,0, 'R-squared=%f' % r_squared_scores[max_r_squared] )


plt.scatter(x, y, color='orange', alpha=0.7)  # scatterplot of points
plt.plot(x, predicted, color='blue', linewidth=2) # plot the linear fit
plt.gca().invert_yaxis()
plt.show()
