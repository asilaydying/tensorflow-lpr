from os import listdir
from os.path import isfile, join
import re
import csv
import tensorflow as tf
from PIL import Image
import numpy as np

car_pictures = "C:\\Dropbox (ASURA Technologies)\\ASURA-DEV\\video\\streamer\\MVI_1521\\"

plates_path = "C:\\temp\\plates.csv"

# Define path to TensorBoard log files
logPath = "C:\\temp\\tens"
picture_pixel_size = 112

number_of_picutres = 3000
training_test_ratio = 0.99
number_of_train_data = int(1000*training_test_ratio)


cars = []
carsfilename = []
rows = []


labels = []
with open(plates_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in spamreader:
        #print(', '.join(row))
        rows.append(row)


counter = 0
for f in listdir(car_pictures):
    number= re.findall('\d+', f)[0]
    i = 0
    frame = int(number);
    carsfilename.append(f)
    while (frame >= int(rows[i][0]) and (i < len(rows)-1)):
        i = i + 1
    if (i> 0 and int(number) <= int(rows[i-1][1])):
        labels.append([0,1])
    else:
        labels.append([1,0])
    im = Image.open(car_pictures + f)
    im = im.convert("L")
    im = im.resize((picture_pixel_size,picture_pixel_size))
    pixel_values = list(im.getdata())
    cars.append(pixel_values)
    counter = counter + 1
    if (counter > number_of_picutres):
        break

print(len(cars))
print(len(labels))
print(carsfilename[len(carsfilename)-1])



#   Adds summaries statistics for use in TensorBoard visualization.  
#      From https://www.tensorflow.org/get_started/summaries_and_tensorboard
def variable_summaries(var):
   with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# Using Interactive session makes it the default sessions so we do not need to pass sess 
sess = tf.InteractiveSession()


# Define placeholders for MNIST input data
with tf.name_scope("MNIST_Input"):
    x = tf.placeholder(tf.float32, shape=[None, picture_pixel_size*picture_pixel_size], name="x")
    y_ = tf.placeholder(tf.float32, [None, 2], name="y_")  

# change the MNIST input data from a list of values to a 28 pixel X 28 pixel X 1 grayscale value cube
#    which the Convolution NN can use.
with tf.name_scope("Input_Reshape"):
    x_image = tf.reshape(x, [-1,picture_pixel_size,picture_pixel_size,1], name="x_image")
    tf.summary.image('input_img', x_image, 5)

# We are using RELU as our activation function.  These must be initialized to a small positive number 
# and with some noise so you don't end up going to zero when comparing diffs
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# Convelution and Pooling - we do Convelution, and then pooling to control overfitting
def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)

# Define the Model

# 1st Convolution layer
with tf.name_scope('Conv1'):
    # 32 features for each 5X5 patch of the image
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5, 5, 1, 32], name="weight")
        variable_summaries(W_conv1)
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32], name="bias")
        variable_summaries(b_conv1)
    # Do convolution on images, add bias and push through RELU activation
    conv1_wx_b = conv2d(x_image, W_conv1,name="conv2d") + b_conv1
    tf.summary.histogram('conv1_wx_b', conv1_wx_b)
    h_conv1 = tf.nn.relu(conv1_wx_b, name="relu")
    tf.summary.histogram('h_conv1', h_conv1)
    # take results and run through max_pool
    h_pool1 = max_pool_2x2(h_conv1, name="pool")

# 2nd Convolution layer
with tf.name_scope('Conv2'):
# Process the 32 features from Convolution layer 1, in 5 X 5 patch.  return 64 features weights and biases
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([5, 5, 32, 64], name="weight")
        variable_summaries(W_conv2)
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([64], name="bias")
        variable_summaries(b_conv2)
    # Do convolution of the output of the 1st convolution layer.  Pool results 
    conv2_wx_b = conv2d(h_pool1, W_conv2, name="conv2d") + b_conv2
    tf.summary.histogram('conv2_wx_b', conv2_wx_b)
    h_conv2 = tf.nn.relu(conv2_wx_b, name="relu")
    tf.summary.histogram('h_conv2', h_conv2)
    h_pool2 = max_pool_2x2(h_conv2, name="pool")

with tf.name_scope('FC'):
    # Fully Connected Layer
    size_after_pool2 = int((picture_pixel_size/4) * (picture_pixel_size/4))
    W_fc1 = weight_variable([size_after_pool2 * 64, 1024], name="weight")
    b_fc1 = bias_variable([1024], name="bias")
    #   Connect output of pooling layer 2 as input to full connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, size_after_pool2*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="relu")

# dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # get dropout probability as a training input.
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope("Readout"):
# Readout layer
    W_fc2 = weight_variable([1024, 2], name="weight")
    b_fc2 = bias_variable([2], name="bias")

# Define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss measurement
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# loss optimization
with tf.name_scope("loss_optimizer"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    # What is correct
    prediction = tf.argmax(y_conv,1)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # How accurate is it?
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy_scl", cross_entropy)
tf.summary.scalar("training_accuracy", accuracy)

# TB - Merge summaries 
summarize_all = tf.summary.merge_all()

# Initialize all of the variables
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

# TB - Write the default graph out so we can view it's structure
tbWriter = tf.summary.FileWriter(logPath, sess.graph)

# Train the model
import time

#  define number of steps and how often we display progress
num_steps = 250
display_every = 5

# Start timer
start_time = time.time()
end_time = time.time()
for i in range(num_steps):
    _, summary = sess.run([train_step, summarize_all], feed_dict={x: cars[:number_of_train_data], y_: labels[:number_of_train_data], keep_prob: 0.5})
    tbWriter.add_summary(summary,i)

    # Periodic status display
    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:cars[:number_of_train_data], y_: labels[:number_of_train_data], keep_prob: 1.0})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))
         
        tbWriter.add_summary(summary,i)
save_path = saver.save(sess, "c:\\temp\\model.ckpt")
#saver.restore(sess, "c:\\temp\\model.ckpt")
# Display summary 
#     Time to train
end_time = time.time()
#print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))
#     Accuracy on test data
testcars = cars[number_of_train_data:]
testlabels = labels[number_of_train_data:]
carsfilename = carsfilename[number_of_train_data:]
for i in range(len(testcars)-1):
    res = prediction.eval(feed_dict={x: testcars[i:i+1], y_: testlabels[i:i+1], keep_prob: 1.0})  
    if (res != testlabels[i:i+1][0][1]):
         print("Wrong prediction, filename: {0}, preditction: {1}".format(carsfilename[i:i+1], res[0]))


print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: cars[number_of_train_data:], y_: labels[number_of_train_data:], keep_prob: 1.0})*100.0))







