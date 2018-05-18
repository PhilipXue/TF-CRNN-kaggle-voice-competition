import tensorflow as tf
import argparse
from keras_model.models import ResNetModel, InceptNetModel
from config import data_root
from tf_dataset import training_valid_dataset
from keras import backend as K


def model_func(features, labels, mode, params):
    keras_model_mapper = {
        'resnet': ResNetModel,
        'inceptnet': InceptNetModel
    }
    tf_model_mapper = {}
    model_name = params['model_name']
    use_keras = model_name in keras_model_mapper
    if use_keras:
        K.set_learning_phase(1)
        model = keras_model_mapper[model_name]
    else:
        model = tf_model_mapper[model_name]
    pre_logits = model().build(features)
    logits = tf.layers.dense(pre_logits, 31, activation=None)
    predicted_class = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        if use_keras:
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
        labels=labels, predictions=predicted_class)}
    return tf.estimator.EstimatorSpec(
        mode, loss=total_loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-m', '--model_name', type=str, default='resnet')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-t', '--train_steps', type=int, default=500000)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--warm_start', type=str, default=None)

    args = parser.parse_args()
    tf.logging.set_verbosity(args.verbose)
    model_path = data_root / 'models' / args.model_name
    model_path.mkdir(exist_ok=True, parents=True)

    estimator_config = tf.estimator.RunConfig(
        model_dir=model_path, save_summary_steps=50, log_step_count_steps=1000)
    estimator_params = {
        'model_name': args.model_name
    }
    estimator = tf.estimator.Estimator(
        model_fn=model_func, config=estimator_config, params=estimator_params)
    dataset = training_valid_dataset(args.batch_size, args.image_size)
    # experiment = tf.contrib.learn.Experiment(
    #     estimator, dataset.training_input_fn, dataset.test_input_fn, train_steps=args.train_steps)
    # experiment.train_and_evaluate()
    train_spec = tf.estimator.TrainSpec(
        dataset.training_input_fn, args.train_steps)
    eval_spec = tf.estimator.EvalSpec(dataset.test_input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
