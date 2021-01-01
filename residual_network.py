import os
import tensorflow.compat.v1 as tf  # noqa
import random
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()

y = 0.0005
num_episodes = 10000
memory = []
memory1 = []
input_dim = 4
output_dim = 4
shape = [256, 64, 16]

input_ = tf.placeholder(tf.float32, [None, input_dim])

W = []
b = []
hidden_outputs = [input_]

if shape:
    for i in range(len(shape)):
        for j in range(2):
            j_in = input_dim if j == 0 else shape[0]
            j_out = input_dim if j == 1 else shape[0]
            W.append(tf.Variable(tf.random_normal([j_in, j_out], stddev=0.03)))
            b.append(tf.Variable(tf.random_normal([j_out])))
            if j == 0:
                hidden_outputs.append(tf.nn.relu(tf.add(tf.matmul(hidden_outputs[2*i], W[2*i]), b[2*i])))
            else:
                hidden_outputs.append(tf.nn.relu(tf.add(tf.add(tf.matmul(hidden_outputs[2*i+1], W[2*i+1]), b[2*i+1]), hidden_outputs[0])))
W.append(tf.Variable(tf.random_normal([input_dim, output_dim], stddev=0.03)))
b.append(tf.Variable(tf.random_normal([output_dim])))
Qout = tf.add(tf.matmul(hidden_outputs[-1], W[-1]), b[-1])
sig_out = tf.sigmoid(Qout)

Q_target = tf.placeholder(shape=(None, output_dim), dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Q_target - Qout))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=y).minimize(loss)

init_op = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    for epoch in range(num_episodes):
        if epoch/(num_episodes/10) == np.floor(epoch/(num_episodes/10)):
            print(str(100*epoch/num_episodes) + "% completed training")
        states = [0, 0, 0, 0]
        for turn in range(4):
            output = np.argmax(sess.run(sig_out, feed_dict={input_: [states]}))  # run states through
            states[output] = 1 if states[output] == 0 else 0
            if states != [1, 1, 1, 1]:
                reward = []
                c = 0
                for h in states:
                    reward.append(-(h - 1))
                _, cost = sess.run([optimiser, loss], feed_dict={input_: [states], Q_target: [reward]})
            if turn == 3:
                memory.append(sum(states))
                memory1.append(cost)
    print("The number of perfect runs: " + str(memory.count(4)) + "/" + str(epoch + 1))
    plt.plot(memory1)
    memory = []
    for epoch in range(20):
        states = [0, 0, 0, 0]
        for i in range(4):
            states[i] = random.randint(0, 1)
        if states == [1, 1, 1, 1]:
            states = [0, 0, 0, 0]
        for _ in range(4 - sum(states)):
            output = np.argmax(sess.run(sig_out, feed_dict={input_: [states]}))  # run states through
            states[output] = 1 if states[output] == 0 else 0
            if sum(states) == 4:
                memory.append(epoch)
    print("The number of perfect runs: " + str(len(memory)) + "/" + str(epoch + 1))
    print(memory)
    plt.show()
