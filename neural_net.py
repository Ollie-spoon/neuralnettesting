import tensorflow.compat.v1 as tf  # importing libraries
import random
import numpy as np

tf.disable_v2_behavior()

y = 0.05                            # creating variables
num_episodes = 200
memory = []
shape = [4, 50, 4]                  # neural network shape including input and output layers
W = []
b = []
hidden_outputs = []

input_ = tf.placeholder(tf.float32, [None, shape[0]])

for i in range(len(shape) - 1):               # Creating the neural network
    W.append(tf.Variable(tf.random_normal([shape[i], shape[i + 1]], stddev=0.03)))
    b.append(tf.Variable(tf.random_normal([shape[i + 1]])))
    if i == len(shape) - 2:
        if len(shape) > 2:
            Qout_new = tf.add(tf.matmul(hidden_outputs[i - 1], W[i]), b[i])
            sig_out_new = tf.sigmoid(Qout_new)
        else:
            Qout_new = tf.add(tf.matmul(input_, W[i]), b[i])
            sig_out_new = tf.sigmoid(Qout_new)
    elif i == 0:
        hidden_outputs.append(tf.nn.relu(tf.add(tf.matmul(input_, W[i]), b[i])))
    else:
        hidden_outputs.append(tf.nn.relu(tf.add(tf.matmul(hidden_outputs[i - 1], W[i]), b[i])))


# creating Tensorflow required things for neural networks
Q_target = tf.placeholder(shape=(None, shape[-1]), dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Q_target - Qout_new))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=y).minimize(loss)

init_op = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    
    # This first section is for training the network to play the game
    
    for epoch in range(num_episodes):
        states = [0, 0, 0, 0]
        for _ in range(4):
            output = np.argmax(sess.run(sig_out_new, feed_dict={input_: [states]}))
            states[output] = 1 if states[output] == 0 else 0
            if states != [1, 1, 1, 1]:
                reward = []
                c = 0
                for h in states:
                    reward.append(-(h - 1))
                sess.run([optimiser, loss], feed_dict={input_: [states], Q_target: [reward]})
            if _ == 3:
                memory.append(sum(states))
    print("In training: Perfect runs - " + str(memory.count(4)) + "/" + str(epoch + 1))
    
    # This second section is used to test the network by using a random gamestate 
    # rather than just [0, 0, 0, 0] as the starting position
    
    memory = []
    for epoch in range(20):
        states = [0, 0, 0, 0]
        for i in range(4):
            states[i] = random.randint(0, 1)
        if states == [1, 1, 1, 1]:
            states = [0, 0, 0, 0]
        for _ in range(4 - sum(states)):
            if _ == 0:
                print("states: " + str(states))
            output = np.argmax(sess.run(sig_out_new, feed_dict={input_: [states]}))  
            states[output] = 1 if states[output] == 0 else 0
            if sum(states) == 4:
                memory.append(epoch)
    print("In teting: Perfect runs - " + str(len(memory)) + "/" + str(epoch + 1))
    print(memory)
