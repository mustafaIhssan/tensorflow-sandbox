print "start"
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# init tf
print "Loading Dataset"
# one_hot meaning the lableing is [0,0,0,0,0,1,0,0,0,0] meaning the label is 5 while there is 10 classes
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# the number of nodes in a hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#number of classes + batch size
n_classes = 10
batch_size = 100


# init x, y_ ( input, true output)
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
	hidden_1_layer = {'weight':tf.Variable(tf.randow_normal([784,n_node_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_node_hl1]))}
	
	hidden_2_layer = {'weight':tf.Variable(tf.randow_normal([n_node_hl1,n_node_hl2])),
						  'biases':tf.Variable(tf.random_normal([n_node_hl2]))}
	
	hidden_3_layer = {'weight':tf.Variable(tf.randow_normal([n_node_hl2,n_node_hl3])),
						  'biases':tf.Variable(tf.random_normal([n_node_hl3]))}

	output_layer = {'weight':tf.Variable(tf.randow_normal([n_node_hl3,n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']) + hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']) + hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']) + hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weight']) + output_layer['biases']

	return output