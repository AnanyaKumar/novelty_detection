"""Train novelty detector on one dataset, and test on another dataset."""

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
import read_notmnist
import numpy as np
from scipy.stats import percentileofscore

import data_processing
import models


def rejection_rates_two_datasets(in_data, ood_data, model, train_epochs=3, train_fraction=1.0):
    (x_train, y_train), (x_valid, y_valid) = in_data
    print(x_train.shape)
    x_ood, y_ood = ood_data
    x_train = x_train[:int(train_fraction * len(x_train))]
    y_train = y_train[:int(train_fraction * len(y_train))]
    # Process data.
    x_train = data_processing.sphere_normalize(x_train)
    x_valid = data_processing.sphere_normalize(x_valid)
    x_ood = data_processing.sphere_normalize(x_ood)

    model.train(x_train, y_train, train_epochs)
    accuracy.validate(x_valid, y_valid)
    confidences, auroc = model.predict(x_ood)
    id_rejection_rates = [0.5, 2.0, 5.0, 10.0, 30.0]
    ood_rejection_rates = [percentileofscore(confidences, r) for r in id_rejection_rates]
    return np.array(ood_rejection_rates), auroc, accuracy


def main():
    num_trials = 2
    train_epochs = 10
    num_labels = 10
    metrics = {
        'rejection_rates': [],
        'accuracies': [],
        'aurocs': []
    }
    for _ in range(num_trials):
        architecture = models.hendrycks_mnist_model(num_labels)
        model = models.SoftmaxDetector(architecture)
        # def batch_flatten(x):
        #     return x.reshape(x.shape[0], -1)
        # model = models.NearestNeighborDetector(batch_flatten)
        # _, ood_data = fashion_mnist.load_data()
        _, _, valid_dataset, valid_labels, _, _ = read_notmnist.get_notMNISTData('.')
        valid_dataset = valid_dataset + 0.5
        valid_dataset = np.reshape(valid_dataset, (valid_dataset.shape[0], 28, 28))
        ood_data = (valid_dataset, valid_labels)
        cur_rejection_rates, auroc, accuracy = rejection_rates_two_datasets(
            mnist.load_data(), ood_data, model, train_epochs, train_fraction=1.0)
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
