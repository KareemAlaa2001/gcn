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
        # v_cost, acc, duration, v_prec, v_recall, v_f1 = evaluate(features, support, y_val, val_mask, placeholders,sess,model)
        v_cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders,sess,model)
        cost_val.append(v_cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(v_cost),
            "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders,sess,model)
    # test_cost, test_acc, test_duration, precision, recall, f1 = evaluate(features, support, y_test, test_mask, placeholders,sess,model)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    test(features, support, y_test, test_mask, placeholders, sess, model)

    return features, support, y_test, test_mask, placeholders, sess, model # returning the objects needed to run a test

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders, sess, model):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.precision, model.recall, model.f1], feed_dict=feed_dict_val)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs, model.labels, model.mask], feed_dict=feed_dict_val)
    # preds = outs_val[2]
    # labels = outs_val[3]
    # mask = outs_val[4]

    # preds_ints = tf.argmax(preds, 1)
    # # print(preds_ints)
    # # print(tf.shape(preds_ints).eval(session=sess))
    # labels_ints = tf.argmax(labels, 1)
    # # print(tf.shape(labels_ints).eval(session=sess))
    # # print(labels_ints)
    # mask = tf.cast(mask, dtype=tf.float32)
    # # print(tf.shape(mask).eval(session=sess))
    # # print(mask)
    # precision = tf.metrics.precision(labels_ints, preds_ints, weights=mask)
    # print("Precision=",precision)

    return outs_val[0], outs_val[1], (time.time() - t_test)
    # return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4]

def test(features, support, labels, mask, placeholders, sess, model):
    feed_dict = construct_feed_dict(features, support, labels, mask, placeholders)
    outs = sess.run([model.outputs, model.labels, model.mask], feed_dict=feed_dict)
    output_list = outs[0]
    label_list = outs[1] # lmao this is dumb im already passing in the labels and mask
    test_mask = outs[2]

    # print(len(list(filter(lambda x: x != 0, output_list))))
    # print(len(list(filter(lambda x: x != 0, label_list))))
    # print(type(output_list))
    print(len(label_list))
    # print(type(test_mask))
    print(len(list(filter(lambda x: any(x), label_list))))
    print(type(labels))
    print(len(labels))
    # print(type(test_mask))
    print(len(list(filter(lambda x: any(x), labels))))
    print(type(mask))
    print(len(mask))
    # print(type(test_mask))
    # print(len(list(filter(lambda x: any(x), mask))))


if __name__ == '__main__':
    train_gcn(False, dataset_string="cora")