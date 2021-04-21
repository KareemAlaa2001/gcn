from __future__ import division
from __future__ import print_function

import time
# from dataHandler import GCNRunner
from lib.gcn.gcn.utils import *
# import tensorflow
import tensorflow.compat.v1 as tf

# from utils import *
from lib.gcn.gcn.models import GCN, MLP

def train_gcn(dsIsFromMemory, gcnRunner=None, 
    dataset_string=None, learning_rate=0.01, 
    num_epochs=200, hidden1=16, dropout=0.5,
    weight_decay=5e-4, early_stopping=10):
    
    if dsIsFromMemory:
        if not gcnRunner:
            raise ValueError("The memory dataset flag was set to true but there was no gcnRunner instance passed in!")

        # if not isinstance(gcnRunner, GCNRunner):
        #     raise ValueError("The passed in object for gcnrunner is not an instance of GCNRunner! ")
    
    else:
        if not dataset_string:
            raise ValueError("The memory-loaded dataset flag was set to false but there was no dataset string passed in for file parsing!")

    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    tf.disable_eager_execution()


    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    del_all_flags(FLAGS)
    
    flags.DEFINE_string('dataset', dataset_string, 'Dataset string.')  # 'cora', 'citeseer', 'pubmed' - original tkipf strings
    flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.') # used to be 0.01
    flags.DEFINE_integer('epochs', num_epochs, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', hidden1, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', dropout, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', weight_decay, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', early_stopping, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    if dsIsFromMemory:
        # x, y, tx, ty, allx, ally, graph, indices
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_custom_data(
            gcnRunner.fvHandler.train_labelled, 
            gcnRunner.fvHandler.train_labels, 
            gcnRunner.fvHandler.test_instances, 
            gcnRunner.fvHandler.test_labels, 
            gcnRunner.fvHandler.train_all, 
            gcnRunner.fvHandler.labels_all_train,
            gcnRunner.adj_graph, 
            gcnRunner.test_indices)
    else:
    # Load data
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders,sess,model)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
            "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders,sess,model)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    return features, support, y_test, test_mask, placeholders, sess, model # returning the objects needed to run a test

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders, sess, model):
    t_test = time.time()
    # print(type(preds), type(labels))
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

if __name__ == '__main__':
    train_gcn(False, dataset_string="cora")