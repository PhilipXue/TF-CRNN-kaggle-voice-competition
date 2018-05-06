import tensorflow as tf
from keras_model.models import ResNetModel
from config import data_root

# TODO: add a mapper from model name to model function


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


def train_estimator(model_name, batch_size, train_steps, image_size):
    model_path = data_root / 'models' / model_name
    model_path.mkdir(exist_ok=True, parents=True)
    estimator_config = tf.estimator.RunConfig(
        model_dir=model_dir, save_summary_steps=50, log_step_count_steps=1000)
    estimator = tf.estimator.Estimator(
        model_fn=model_func, config=estimator_config)
    experiment = tf.contrib.learn.Experiment(
        model_estimator, dataset.train_input_fn, dataset.test_input_fn, train_steps=train_steps)
    experiment.train_and_evaluate()

if __name__ == "__miain__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-m', '--model_name', type=str, default='test')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-t', '--train_steps', type=int, default=500000)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--warm_start', type=str, default=None)

    args = parser.parse_args()
    tf.logging.set_verbosity(args.verbose)
    train_estimator(args.model_name, args.batch_size,
                    args.train_steps, args.image_size)
