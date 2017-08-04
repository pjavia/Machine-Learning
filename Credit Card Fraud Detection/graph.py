import tensorflow as tf
from preprocessing import process


batch = None
mini_batch = 9
learning_rate = 0.00000001
beta1 = 0.09
beta2 = 0.099
linear = tf.Graph()

with linear.as_default():

	train_data = tf.placeholder(tf.float32, [batch, 30])
	label_data = tf.placeholder(tf.float32, [batch])
	batch_size = tf.placeholder(tf.float32, [])
	#keep_prob = tf.placeholder(tf.float32, [])

	layer_1 = tf.contrib.layers.fully_connected(train_data, 2, weights_initializer=tf.zeros_initializer())
	#layer_2 = tf.nn.dropout(tf.contrib.layers.fully_connected(layer_1, 20), keep_prob=keep_prob)
	#layer_3 = tf.contrib.layers.fully_connected(layer_2, 2)
	#layer_4 = tf.contrib.layers.fully_connected(layer_3, 2)
	final = tf.nn.softmax(layer_1)

	probability = tf.reduce_max(final, -1)

	prediction = tf.argmax(final, -1)

	train_accuracy = tf.scalar_mul(batch_size, tf.to_float(tf.count_nonzero(tf.cast(tf.equal(tf.to_int64(label_data), prediction), dtype=tf.int64))))




	loss = tf.reduce_mean(-1.0*((1.0 - label_data)*tf.log(tf.clip_by_value(1.0 - probability, clip_value_min=1e-15, clip_value_max=0.9999999)) + (label_data)*tf.log(tf.clip_by_value(probability, clip_value_min=1e-15, clip_value_max=0.9999999))))

	opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
	train_opt = opt.minimize(loss)


	#train_accuracy = tf.scalar_mul(tf.constant(100, dtype=tf.float32), tf.reduce_mean(tf.to_float(tf.count_nonzero(tf.cast(tf.equal(tf.to_int64(label_data), prediction), dtype=tf.int64)))))
	#train_accuracy = tf.scalar_mul(tf.constant(100.0/float(batch) dtype=tf.float32), tf.to_float(tf.count_nonzero(tf.cast(tf.equal(tf.to_int64(label_data), prediction), dtype=tf.int64))))
	
	#print train_accuracy
	weights_of_model = tf.trainable_variables()

	

	summary_hist_W = tf.summary.histogram('W', weights_of_model[0])
	summary_hist_b = tf.summary.histogram('W', weights_of_model[1])
	summary_loss = tf.summary.scalar('Loss', loss)
	#summary_train_acc = tf.summary.scalar('Training Accuracy', train_accuracy)
	summary_op = tf.summary.merge_all() 



	init = tf.global_variables_initializer()


with tf.Session(graph=linear) as sess:

	sess.run(init)
	train_writer = tf.summary.FileWriter('summary_directory', sess.graph)

	training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels = process()

	
	
	step = 0
	for epoch in range(1, 2):

		for i in range(0, len(training_data) - mini_batch, mini_batch):
			x_feed = []
			y_feed = []
			for j in range(i, i + mini_batch):
				x_feed.append(training_data[j])
				y_feed.append(training_labels[j])

			loss_, _, summary_full = sess.run([loss, train_opt, summary_op], feed_dict={train_data: x_feed, label_data: y_feed, batch_size: 100.0/float(len(x_feed))}) #, keep_prob: 0.5})
			step += 1
			train_writer.add_summary(summary_full, step)
			#acc_ = sess.run(train_accuracy, feed_dict={train_data: training_data, label_data: training_labels, keep_prob: 1})

			print loss_

		x_feed = training_data
		y_feed = training_labels

		training_acc_ = sess.run(train_accuracy, feed_dict={train_data: x_feed, label_data: y_feed, batch_size: 100.0/float(len(x_feed))})

		x_feed = validation_data
		y_feed = validation_labels

		validation_acc_ = sess.run(train_accuracy, feed_dict={train_data: x_feed, label_data: y_feed, batch_size: 100.0/float(len(x_feed))})

		x_feed = testing_data
		y_feed = testing_labels

		testing_acc_ = sess.run(train_accuracy, feed_dict={train_data: x_feed, label_data: y_feed, batch_size: 100.0/float(len(x_feed))})


		print 'Training Accuracy :', training_acc_ 
		print 'Validation Accuracy :', validation_acc_
		print 'Testing Accuracy :', testing_acc_








