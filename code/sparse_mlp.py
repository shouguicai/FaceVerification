"""by CAICAI.
"""
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import scipy.io
import sys
import os
import time

data_path = './datasets/save_mat/emb_mat_10k.mat'
data_path_test = './datasets/save_mat/emb_mat_10k_test.mat'
logs_dir = './logs/sparse_logs/'
model_dir = './models/sparse_models/'
step_max = 20000
D = 128
M = 500
K = 10575
next_batch_size = 5000
dropout = 0.8
learning_rate = 1e-2


# Import data
def read_data(data_path):
    return scipy.io.loadmat(data_path)
    
def l21_norm_matrix(W):
    # Computes the L21 norm of a symbolic matrix W
    return tf.reduce_sum(tf.norm(W, axis=1))
def l21_norm_vector(W):
    # Computes the L21 norm of a symbolic matrix W
    return tf.reduce_sum(tf.norm(W))
def l1_norm(W):
    # Computes the L1 norm of a symbolic matrix W
    return tf.reduce_sum(tf.abs(W))
    
def train():
    # read data
    data = read_data(data_path)
    data_test = read_data(data_path_test)
    # training data
    batch_xs = data['emb_array']
    label_list = data['labels']
    num_images = batch_xs.shape[0]
    batch_ys = np.zeros((num_images, K))
    for ii in range(num_images):   # transfer label list to label matrix
        batch_ys[ii][label_list[0][ii]] =1

    # test data
    test_xs = data_test['emb_array']
    label_list_test = data_test['labels']
    num_images_test = test_xs.shape[0]
    test_ys = np.zeros((num_images_test, K))
    for ii in range(num_images_test):   # transfer label list to label matrix
        test_ys[ii][label_list_test[0][ii]] =1

    sess = tf.InteractiveSession()
    
    def next_batch(num, data, labels):
        '''
        Return a total of `num` random samples and labels. 
        '''
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        labels_shuffle = [labels[ i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, D], name='x-input')
        y_desired = tf.placeholder(tf.float32, [None, K], name='y-input')
        sparsity_constraint = tf.placeholder(tf.float32, name='sparsity_constraint')
    
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                #tf.add_to_collection('vars', weights)
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                #tf.add_to_collection('vars', biases)
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            average_density = tf.reduce_mean(tf.reduce_sum(tf.cast((activations > 0), tf.float32), axis=[1]))
            tf.summary.scalar('AverageDensity', average_density)
            return activations,weights,biases

    hidden1,W1,B1 = nn_layer(x, D, M, 'hidden',act= tf.nn.relu)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob,name='dropped')

    # Do not apply softmax activation yet, see below.
    y_hat,W2,B2 = nn_layer(dropped, M, K, 'output', act=tf.identity)
    
    y_hat_softmax = tf.nn.softmax(y_hat,name='y_hat_softmax')
    
    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.
        epsilon = 1e-7 # After some training, y can be 0 on some classes which lead to NaN 
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_desired,logits=(y_hat+epsilon),name='diff')
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff,name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    
    with tf.variable_scope('Loss'):

        loss = tf.add(cross_entropy ,sparsity_constraint *(l1_norm(hidden1)+l1_norm(y_hat_softmax)+ l21_norm_matrix(W1)+  l21_norm_matrix(W2)  + l21_norm_vector(B1)+l21_norm_vector(B2)),name='loss')

    tf.summary.scalar('loss', loss) # Graph the loss

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_desired, 1),name='correct_prediction')
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
    tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()
    
    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries
    
    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            #xs = batch_xs
            #ys = batch_ys
            xs,ys = next_batch(next_batch_size,batch_xs,batch_ys)
            k = dropout
        else:
            #xs = test_xs
            #ys = test_ys
            xs,ys = next_batch(next_batch_size,test_xs,test_ys)
            k = 1.0
        return {x: xs, y_desired: ys, keep_prob: k, sparsity_constraint: sc}

            
    for sc in [0]:
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        result_folder = subdir + '-fc-sc' + str(sc)
        #model_folder = model_dir + subdir
        model_name = 'my-model'+ '-fc-sc' + str(sc)
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_dir + result_folder + '/train',sess.graph)
        test_writer = tf.summary.FileWriter(logs_dir + result_folder + '/test',sess.graph)

        config = tf.ConfigProto(device_count={"CPU": 50}, # limit to num_cpu_core CPU usage  
                inter_op_parallelism_threads = 5,   
                intra_op_parallelism_threads = 5,  
                log_device_placement=True)  

        sess.run(tf.global_variables_initializer())  
        #saver.save(sess, model_dir + model_folder)
    
        startTime = time.time()

        for step in range(step_max):
            
            if step % 10 == 0:  # Record summaries and test-set accuracy
                summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                test_writer.add_summary(summary, step)
                print('Accuracy at step %s for test set: %s' % (step, acc))

            else:  # Record train set summaries, and train
            
                if step % 100 == 99:  # Record execution stats
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary,acc,_ = sess.run([merged,accuracy,train_step],
                                            feed_dict=feed_dict(True),
                                            options=run_options,
                                            run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(summary, step)
                    #print('Accuracy at step %s for training set: %s' % (step, acc))
                    #print('Adding run metadata for', step)

                else:  # Record a summary
                    summary, _ = sess.run([merged,train_step], feed_dict=feed_dict(True))
                    train_writer.add_summary(summary, step)
            if step % 100 == 0:
                subdir1 = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
                model_folder = model_dir + subdir1
                tf.gfile.MkDir(model_folder)
                saver.save(sess,model_folder+'/'+model_name,global_step=step)
        train_writer.close()
        test_writer.close()

        duration = time.time() - startTime
        print("Training time taken: %f" % (duration))
    

def main(_):
    if tf.gfile.Exists(logs_dir + '/train'):
        tf.gfile.DeleteRecursively(logs_dir)
    tf.gfile.MakeDirs(logs_dir)
    tf.gfile.MkDir(logs_dir + '/train')
    train()

if __name__ == '__main__':
  tf.app.run(main=main)
  
  
#tensorboard --logdir='./logs/sparse_logs/'