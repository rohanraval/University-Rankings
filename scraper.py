import csv

def getData(input_var):
	dataset = {}
	"""
	with open('cwurData.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			dependent = 'world_rank'
			independent = 'publications'
			if row[independent] != '' and row[dependent] != '':
				dataset[ int(row[independent]) ] = int(row[dependent])
	"""
	with open('timesData.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		dependent = 'world_rank'
		independent = input_var
		tie = False
		for row in reader:
			x = row[independent]
			y = row[dependent]
			#if "=" in y:
			#	tie = True
			#else:
			#	tie = False

			if x != '' and y != '' and "-" not in y and row['year'] == '2016':
					#if tie is True:
					#	y = int(y[1:])+1
					if "=" in y:
						y = y[1:]
					dataset[ float(x) ] = int(y)

	return dataset