import csv

def process():

	file = open('creditcard.csv', 'rb')
	reader = csv.reader(file)

	clean = []

	for row in reader:

		clean.append(row)

	del clean[0]

	data = []
	label = []
	for sample in clean:

		consig = map(float, sample)
		label.append(consig[-1])
		del consig[-1]
		data.append(consig)

	# Divide data into 3:2:1 that is 6x = len(data) i.e. int(len(data)/6.0)

	x = int(len(data)/6.0)

	train_data = data[0:3*x]
	train_labels = label[0:3*x]

	val_data = data[3*x:5*x]
	val_labels = label[3*x:5*x]

	test_data = data[5*x:]
	test_labels = label[5*x:]


	return train_data, train_labels, val_data, val_labels, test_data, test_labels

process()










	

