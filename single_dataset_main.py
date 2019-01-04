"""Test novelty detector on a single dataset, by splitting on labels."""

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import numpy as np
from scipy.stats import percentileofscore

import data_processing
import models


def rejection_rates_single_dataset(dataset, num_in_domain_labels, label_map, model, train_epochs=3,
                                   train_fraction=1.0):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train = x_train[:int(train_fraction * len(x_train))]
    y_train = y_train[:int(train_fraction * len(y_train))]
    # Process data.
    x_train = data_processing.force_to_0_1(x_train)
    x_test = data_processing.force_to_0_1(x_test)
    y_train = np.array(map(label_map, y_train))
    y_test = np.array(map(label_map, y_test))
    # Filter data.
    def in_domain(label):
        return 0 <= label < num_in_domain_labels
    def out_of_domain(label):
        return not in_domain(label)
    xs, ys = {}, {}
    xs['train'], ys['train'] = data_processing.label_filter(x_train, y_train, in_domain)
    xs['valid'], ys['valid'] = data_processing.label_filter(x_test, y_test, in_domain)
    xs['ood'], ys['ood'] = data_processing.label_filter(x_test, y_test, out_of_domain)

    model.train(xs['train'], ys['train'], train_epochs)
    accuracy = model.validate(xs['valid'], ys['valid'])
    confidences, auroc = model.predict(xs['ood'])
    id_rejection_rates = [0.5, 2.0, 5.0, 10.0, 30.0]
    ood_rejection_rates = [percentileofscore(confidences, r) for r in id_rejection_rates]
    return np.array(ood_rejection_rates), auroc, accuracy


def main():
    num_trials = 2
    train_epochs = 5
    num_in_domain_labels = 5
    metrics = {
        'rejection_rates': [],
        'accuracies': [],
        'aurocs': []
    }
    for _ in range(num_trials):
        label_permutation = np.random.permutation(range(10))
        label_map = lambda i: label_permutation[i]
        architecture = models.hendrycks_mnist_model(num_in_domain_labels)
        model = models.SoftmaxDetector(architecture)
        # def mean_func(x):
        #     dim = len(x.shape)
        #     mean = np.mean(x, axis=tuple(range(1, dim)))
        #     print(np.mean(mean))
        #     return np.expand_dims(mean, axis=-1)
        # def batch_flatten(x):
        #     return x.reshape(x.shape[0], -1)
        # model = models.NearestNeighborDetector(mean_func)
        cur_rejection_rates, auroc, accuracy = rejection_rates_single_dataset(
            mnist, num_in_domain_labels, label_map, model, train_epochs, train_fraction=1.0)
        metrics['rejection_rates'].append(cur_rejection_rates)
        metrics['accuracies'].append(accuracy)
        metrics['aurocs'].append(auroc)
    for metric_name in metrics:
        values = metrics[metric_name]
        mean_values = np.mean(values, axis=0)
        std_values = np.std(values, axis=0) / np.sqrt(num_trials)
        print(metric_name)
        print('mean: ', mean_values)
        print('std-dev: ', std_values)
    print(metrics)


if __name__ == "__main__":
    main()
