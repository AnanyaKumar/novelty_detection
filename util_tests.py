import unittest
from models import compute_auroc

class TestStringMethods(unittest.TestCase):

    def test_compute_auroc(self):
        valid_scores = [1, 2, 3, 4, 5]
        ood_scores = [0, 0, 0, 1.5, 2.5]
        self.assertAlmostEqual(compute_auroc(valid_scores, ood_scores), 0.88)

    def test_compute_auroc_bounds(self):
        valid_scores = [1, 2, 3, 4, 5]
        ood_scores = [0, 0, 0, 1.5, 6.0]
        self.assertAlmostEqual(compute_auroc(valid_scores, ood_scores), 0.76)


if __name__ == '__main__':
    unittest.main()