from evalHelpers import calc_precision, calc_recall
import tensorflow.compat.v1 as tf
import keras, numpy

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    # print(preds)
    # print(labels)
    # print(mask)
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


# skeleton function to not throw an error ( at least i think it wont)
# trying to just return int versions of the predictions
def masked_precision(preds, labels, mask):
    pass
    # preds_ints = tf.argmax(preds, 1)
    # labels_ints = tf.argmax(labels, 1)
    # mask = tf.cast(mask, dtype=tf.float32)

    # tf.math.confusion_matrix(labels_ints, preds_ints, num_classes=2, weights=mask)
    # # precision = tf.metrics.precision(labels_ints, preds_ints, weights=mask)
    # print(preds_ints)
    # print(labels_ints)
    # print(mask)



    # return precision

def masked_recall(preds, labels, mask):
    pass


def masked_f1_score(preds, labels, mask):
    pass

# def masked_precision(preds, labels, mask):
#     print(preds)
#     print(labels)

#     preds_ints = tf.argmax(preds, 1)
#     labels_ints = tf.argmax(labels, 1)
    
#     mask = tf.cast(mask, dtype=tf.float32)
#     print(mask)

#     trueposlayer = keras.metrics.TruePositives()
#     trueposlayer.update_state(labels_ints, preds_ints, sample_weight=mask)
#     truepos = trueposlayer.result()
    
#     falseposlayer = keras.metrics.FalsePositives()
#     falseposlayer.update_state(labels_ints,preds_ints, sample_weight=mask)
#     falsepos = falseposlayer.result()

#     return calc_precision(truepos, falsepos)

# def masked_recall(preds, labels, mask):
#     preds_ints = tf.argmax(preds, 1)
#     labels_ints = tf.argmax(labels, 1)

#     mask = tf.cast(mask, dtype=tf.float32)

#     trueposlayer = keras.metrics.TruePositives()
#     trueposlayer.update_state(labels_ints, preds_ints, sample_weight=mask)
#     truepos = trueposlayer.result()
    
#     falseneglayer = keras.metrics.FalseNegatives()
#     falseneglayer.update_state(labels_ints,preds_ints, sample_weight=mask)
#     falseneg = falseneglayer.result()

#     return calc_recall(truepos, falseneg)

# def masked_f1_score(preds, labels, mask):
#     preds_ints = tf.argmax(preds, 1)
#     labels_ints = tf.argmax(labels, 1)

#     mask = tf.cast(mask, dtype=tf.float32)

#     trueposlayer = keras.metrics.TruePositives()
#     trueposlayer.update_state(labels_ints, preds_ints, sample_weight=mask)
#     truepos = trueposlayer.result()
    
#     falseposlayer = keras.metrics.FalsePositives()
#     falseposlayer.update_state(labels_ints,preds_ints, sample_weight=mask)
#     falsepos = falseposlayer.result()

#     falseneglayer = keras.metrics.FalseNegatives()
#     falseneglayer.update_state(labels_ints,preds_ints, sample_weight=mask)
#     # falseneg = falseneglayer.result().numpy()
#     falseneg = falseneglayer.result()
#     recall = calc_recall(truepos, falseneg)
#     precision = calc_precision(truepos, falsepos)

#     return 2*((precision*recall)/(precision+recall))
#     # return tf.convert_to_tensor(2*((precision*recall)/(precision+recall)))