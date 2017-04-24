# Algorithm for analyzing University Rankings developed by Rohan Raval
# Dataset obtained from: https://www.kaggle.com/mylesoneill/world-university-rankings

import scraper
import numpy as np
import pandas as pd
from scipy import sparse
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

# get the input of the x-parameter that correlates the y-value (rank)
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

# a dictionary of the R-squared values of each model
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

# get the name of the model with the max R-squared and its value
max_r_squared = keys[vals.index(max(vals))]

print "Model = %s" % max_r_squared
print "R-squared = %f" % r_squared_scores[max_r_squared] 


# for the model with the max R-squared, get the predicted values
# on the line of best fit, can now be extended to prediction beyond x-data
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

### data visualization (using matplotlib) ###

# adjust the plot and show labels, title, and R-squared on plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Effect of %s Score on University\'s World Ranking' % input_var.title())
ax.set_xlabel('%s Score' % input_var.title())
ax.set_ylabel('World Ranking')
ax.text(50,5, 'R-squared=%f' % r_squared_scores[max_r_squared] )

# show scatterplot of datapoints
plt.scatter(x, y, color='orange', alpha=0.7)

# plot the fitted linear model
plt.plot(x, predicted, color='blue', linewidth=2)

# invert y-axis because low rank number is actually better (this is more intuitive modeling)
plt.gca().invert_yaxis() 

# show the plot
plt.show()
