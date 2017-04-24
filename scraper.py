# Scraper for scraping University Rankings developed by Rohan Raval
# Dataset scraped from: https://www.kaggle.com/mylesoneill/world-university-rankings

import csv

def getData(input_var):
	dataset = {} # make the dictionary of x:y data points

	# reading in every row of csv file and creating dict of x:y points
	with open('timesData.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		dependent = 'world_rank'
		independent = input_var

		for row in reader:
			x = row[independent]
			y = row[dependent]

			if ',' in x:
				x = x.replace(',', '')

			if x != '' and y != '' and "-" not in y and row['year'] == '2016':
				if "=" in y: # if there is a ranking tie, the cell has a '=' in front, so get rid of that 
					y = y[1:]
				dataset[ float(x) ] = int(y)

	return dataset