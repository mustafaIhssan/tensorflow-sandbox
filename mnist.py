import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# init tf
print "init - downloading Dataset"
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
print "finish downlaoding"


# 784 = the input pixels 
# 10  = the output (0-9)
print "placeholder created"
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# perpare: init values of the w and b with tensors of zeros  
# w = weight
# b = bias
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# exeute: init valuse
print "init the values of the networks"
sess.run(tf.initialize_all_variables())

# regression model
# y = output of the neuron
# x = the value of the pixel
y = tf.matmul(x,W) + b

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# Training 
print "start training"
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
	if(i%75==0):
		print "{}% complete.".format((i/1000.0)*100)
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
print "done trining"

# Evalute the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print "Done !!"
