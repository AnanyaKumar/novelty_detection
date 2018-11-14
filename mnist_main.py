"""Simple novelty detector for MNIST."""

from keras.datasets import mnist
import numpy as np
from scipy.stats import percentileofscore

import data_processing
import models



def get_rejection_rates(in_domain_labels, architecture, train_epochs=3):
    # Get and process data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    def in_domain(label):
        return label in in_domain_labels
    def out_of_domain(label):
        return not in_domain(label)
    xs, ys = {}, {}
    xs['train'], ys['train'] = data_processing.label_filter(x_train, y_train, in_domain)
    xs['valid'], ys['valid'] = data_processing.label_filter(x_test, y_test, in_domain)
    xs['ood'], ys['ood'] = data_processing.label_filter(x_test, y_test, out_of_domain)
    for key in xs:
        xs[key] = data_processing.force_to_0_1(xs[key])

    model = models.SoftmaxDetector(architecture)
    model.train(xs['train'], ys['train'], train_epochs)
    model.validate(xs['valid'], ys['valid'])
    confidences = model.predict(xs['ood'])
    id_rejection_rates = [0.5, 2.0, 5.0, 10.0]
    ood_rejection_rates = [percentileofscore(confidences, r) for r in id_rejection_rates]
    return np.array(ood_rejection_rates)


def main():
    num_trials = 5
    train_epochs = 4
    # TODO: Fix so they don't have to be consecutive.
    in_domain_labels = [0, 1, 2, 3, 4]
    rejection_rates = []
    for _ in range(num_trials):
        architecture = models.linear_model(len(in_domain_labels))
        cur_rejection_rates = get_rejection_rates(in_domain_labels, architecture, train_epochs)
        rejection_rates.append(cur_rejection_rates)
    mean_rejection_rates = np.mean(rejection_rates, axis=0)
    print(mean_rejection_rates)


if __name__ == "__main__":
    main()
