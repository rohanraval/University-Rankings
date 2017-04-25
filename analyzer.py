# Script for analyzing University Rankings developed by Rohan Raval
# Dataset obtained from: https://www.kaggle.com/mylesoneill/world-university-rankings

import sys
import csv
import numpy as np
import pandas as pd
from scipy import sparse
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


def scrapeData_factorsAffectingRank(filename, attribute):

	dataset = {} # make the dictionary of x:y data points

	# reading in every row of csv file and creating dict of x:y points
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile)
		dependent = 'world_rank'

		for row in reader:
			x = row[attribute]
			y = row[dependent]

			# manage case for comma-separated numbers
			if ',' in x:
				x = x.replace(',', '')

			if x != '' and y != '' and "-" not in x and "-" not in y:
				if (filename in ['cwurData.csv', 'shanghaiData.csv'] and row['year'] == '2015') or (filename == 'timesData.csv' and row['year'] == '2016'):
				 	# if there is a ranking tie, the cell has a '=' in front, so get rid of that 
					if "=" in y:
						y = y[1:]
				dataset[ float(x) ] = int(y)

	return dataset

def scrapeData_collegeChange(filename, college_name, attribute):

	dataset = {} # make the dictionary of x:y data points

	# reading in every row of csv file and creating dict of x:y points
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile)

		for row in reader:
			x = row['year']
			y = row[attribute]
			university_name = 'institution' if filename == "cwurData.csv" else 'university_name'

			# manage case for comma-separated numbers
			if ',' in y:
				y = y.replace(',', '')

			if x != '' and y != '' and "-" not in x and "-" not in y and college_name in row[university_name]:
				dataset[ int(x) ] = float(y)

	return dataset

def data_formatting(filename, analysis_type, attribute, college_name=''):

	if analysis_type == '1':
		# get a dictionary of x:y pairs from dataset
		# x = world-ranks
		# y = input attribute, e.g. research score (out of 100)
		data = scrapeData_factorsAffectingRank(filename, attribute)
	else:
		# get a dictionary of x:y pairs from dataset
		# x = years
		# y = input attribute, e.g. research score (out of 100)
		data = scrapeData_collegeChange(filename, college_name, attribute)

	df = pd.DataFrame.from_dict(data.items()) # convert data to a dataframe
	x = df[0].to_frame() # x-values from the dataframe
	y = df[1].to_frame() # y-values from the dataframe

	return { "x":x, "y":y }

def linear_modeling(x,y):

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

	# make a dictionary of the R-squared values of each model
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

	return { 'predicted' : predicted, 'rsquared' : r_squared_scores[max_r_squared] }


### data visualization (using matplotlib) ###
def data_visualization(x,y,predicted,rsquared,attribute,filename, college_name):

	# adjust the plot and show labels, title, and R-squared on plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#fig.subplots_adjust()
	fig.suptitle('What Makes a Good University?', fontsize=14, fontweight='bold')

	if college_name == '':
		ax.set_title('Effect of %s %s on University\'s World Ranking' % (attribute.title(), "Rank" if filename == 'cwurData.csv' else "Score")
		)
		# invert y-axis because low rank number is actually better (this is more intuitive modeling)
		#plt.gca().invert_yaxis() 
		#if filename == 'cwurData.csv':
			#plt.gca().invert_xaxis() # due to cwur file containing rank instead of scores

		ax.set_xlabel('%s %s' % ('Rank' if filename == 'cwurData.csv' else 'Score', attribute.title()) ) 
	
		ax.set_ylabel('World Ranking')

		if filename == 'cwurData.csv':
			ax.text(200,5, 'R-squared=%f' % rsquared )
		else:
			ax.text(50,5, 'R-squared=%f' % rsquared )

	else:
		if filename == 'cwurData.csv':
			plt.gca().invert_yaxis() 

		ax.set_title('Change in %s %s at %s over Time' % (attribute.title(), "Rank" if filename == 'cwurData.csv' else "Score", college_name)
		)

		ax.set_xlabel('Year')
		ax.set_ylabel(attribute)

		ax.text(2014,5, 'R-squared=%f' % rsquared )

	

	# show scatterplot of datapoints
	plt.scatter(x, y, color='orange', alpha=0.7)

	# plot the fitted linear model
	plt.plot(x, predicted, color='blue', linewidth=2)

	# show the plot
	plt.show()


######## MAIN METHOD ########

#get data filename
filename = ''
filename_input = raw_input("Enter data filename: \n%s \n%s \n%s \n" % ("1: CWUR", "2: Times", "3: Shanghai")
)
if filename_input == '1':
	filename = 'cwurData.csv'
elif filename_input == '2':
	filename = 'timesData.csv'
elif filename_input == '3':
	filename = 'shanghaiData.csv'
else:
	print "INVALID INPUT"
	sys.exit(1)

print filename


# get type of analysis to perform
analysis_type = raw_input("Enter type of analysis: \n%s \n%s \n" % 
	("1: Attributes affecting University Rank", "2: Change in Attribute over time for a University")
)

attribute = raw_input("Enter attribute to measure: ")

data = {} # x:y pairs of data 
college_name = ''
if analysis_type == '1':
	data = data_formatting(filename, analysis_type, attribute)
else:
	college_name = raw_input("Enter University Name: ")
	print college_name
	data = data_formatting(filename, analysis_type, attribute, college_name)

linear_model_data = linear_modeling(data['x'], data['y'])

data_visualization(data['x'], data['y'], linear_model_data['predicted'], linear_model_data['rsquared'], attribute, filename, college_name)


