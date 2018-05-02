import tensorflow as tf
from keras_model.models import ResNetModel


def model_func(features, labels, model):
    K.set_learning_phase(1)
    pre_logits = ResNetModel().build(features)
    # logits = tf.layers.Dense(31, activation=None)(pre_logits)
    logits = tf.layers.dense(pre_logits, 31, activation=None)
    predicted_class = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        K.set_learning_phase(0)
        predictions = {
            'class': predicted_class,
            'prob': tf.nn.softmax(logits),
            'embedding': embedding
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    cls_loss = tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(labels, logits))
    total_loss = cls_loss
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=args.learning_rate)
        train_op = optimizer.minimize(
            total_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
    eval_metric_ops = {'accuracy': tf.metrics.accuracy(
        labels=labels, predictions=predicted_class)
    return tf.estimator.EstimatorSpec(
        mode, loss=total_loss, eval_metric_ops=eval_metric_ops)


def train_estimator(model, batch_size, train_steps, image_size):
    experiment = tf.contrib.learn.Experiment(
        model_estimator, dataset.train_input_fn, dataset.test_input_fn, train_steps=train_steps)
    experiment.train_and_evaluate()
