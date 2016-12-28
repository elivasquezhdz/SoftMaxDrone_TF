import tensorflow.python.platform
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join

NUM_LABELS = 2
BATCH_SIZE = 10
num_epochs = 10

def extract_test():
	list = []
	path = 'featsEdg\\'
	featlist = [f for f in listdir(path) if isfile(join(path, f))]
	for i in range(len(featlist)):
		list.append(np.load(path + featlist[i]))
	return list

def extract_data(): #Reads the image vectors, appends a dummy class and returns the lables for each case
	data = np.load('vectors_EDGE.npy') # Read the extracted features
	black = np.zeros(data.shape,dtype=np.float32) #Create dummy class
	fvects = np.vstack((data,black)) #Join matrices
	labels = [] #List for out labels
	for i in range(data.shape[0]): #Append 1 for first class
		labels.append(1)
	for i in range(black.shape[0]): #Apeend 0 for dummy class
		labels.append(0)
	labels_np = np.array(labels).astype(dtype=np.uint8) #Create numpy array of labels
	labels_onehot = (np.arange(NUM_LABELS) == labels_np[:,None]).astype(np.float32) #Convert to one hot coded matrix
	return fvects,labels_onehot
def main(_):
	testz = np.matrix(np.zeros(900,dtype=np.float32))
	testo = np.matrix(np.ones(900,dtype=np.float32))
	
	train_data,train_labels = extract_data()
	test = extract_test()#np.load(testf),testf = 'test.npy'
	#print(test)
	train_size,num_features = train_data.shape
	x = tf.placeholder("float", shape=[None, num_features])
	y_ = tf.placeholder("float", shape=[None, NUM_LABELS])
	W = tf.Variable(tf.zeros([num_features,NUM_LABELS]))
	b = tf.Variable(tf.zeros([NUM_LABELS]))
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	a = []
	with tf.Session() as s:
		tf.initialize_all_variables().run()	
		#tf.global_variables.initializer().run()
		for step in range(num_epochs*train_size//BATCH_SIZE):
			#print (step)
			offset = (step * BATCH_SIZE) % train_size
			batch_data = train_data[offset:(offset + BATCH_SIZE), :]
			batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
			train_step.run(feed_dict = {x: batch_data, y_ : batch_labels})
		print
		print ('Weight matrix.')
		print (s.run(W))
		a = s.run(W)
		print
		print ('Bias vector.')
		print
		print (s.run(b))
		print
		print ("Applying model to first test instances.")
		first = train_data[:1]
		#print ("Point =", first)
		print ("Wx +b = ", s.run(tf.matmul(first,W)+b))
		print ("softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(first,W)+b)))
		print
		print ("Accuracy: ", accuracy.eval(feed_dict={x: train_data, y_: train_labels}))
		print
		print ("Test Zeros:")
		print ("Wx +b = ", s.run(tf.matmul(testz,W)+b))
		print ("softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(testz,W)+b)))
		print
		print ("Test Ones:")
		print ("Wx +b = ", s.run(tf.matmul(testo,W)+b))
		print ("softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(testo,W)+b)))
		for i in range(len(test)):
			print ('Test:' + str(i))
			print ("Wx +b = ", s.run(tf.matmul(test[i],W)+b))
			print ("softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(test[i],W)+b)))
		
		
		
if __name__ == '__main__':
	tf.app.run(main = main )