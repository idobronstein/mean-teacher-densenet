"""Train ConvNet Mean Teacher on CIFAR-10 training set and evaluate against a validation set

This runner converges quickly to a fairly good accuracy.
On the other hand, the runner experiments/cifar10_final_eval.py
contains the hyperparameters used in the paper, and converges
much more slowly but possibly to a slightly better accuracy.
"""

import logging
from datetime import datetime

from experiments.run_context import RunContext
from datasets import Cifar100ZCA
from mean_teacher.model import Model
from mean_teacher import minibatching


logging.basicConfig(filename='log.txt', filemode='w', level=logging.INFO)
LOG = logging.getLogger('main')


def run(data_seed=0):
    n_labeled = 10000

    model = Model(RunContext(__file__, 0))
    model['flip_horizontally'] = True
    model['normalize_input'] = False  # Keep ZCA information
    model['rampdown_length'] = 0
    model['rampup_length'] = 5000
    model['training_length'] = 80000
    model['max_consistency_cost'] = 3000.0
    model['student_dropout_probability'] = 0.8
    model['teacher_dropout_probability'] = 0.8

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    cifar = Cifar100ZCA(data_seed, n_labeled)
    training_batches = minibatching.training_batches(cifar.training, n_labeled_per_batch=50)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(cifar.evaluation)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    run()
