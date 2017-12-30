import tensorflow as tf

#initialize data
X = [2, 5, 7]
Y = [3, 4, 6]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis H = WX+b
H = W * X + b

#cost/loss function
cost = tf.reduce_mean(tf.square(H - Y))

#minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#launch the graph in a session
sess = tf.Session()
#initialize global variables in the graph
sess.run(tf.global_variables_initializer())

#output
for epoch in range(1001):
    sess.run(train)
    if (epoch % 100 == 0):
        print (("step %4dth : cost = %f, W = %f, b = %f")
               % (epoch, sess.run(cost), float(sess.run(W)), float(sess.run(b))))
