import unittest

import ucdd_eval


class MyTestCase(unittest.TestCase):
    def test_fpr_latency_no_drift_detected(self):
        self.assertEqual((0, 1, False), ucdd_eval.fpr_and_latency_when_averaging([], 7, 2))

    def test_fpr_latency_only_exact_detection(self):
        self.assertEqual((0, 0, True), ucdd_eval.fpr_and_latency_when_averaging([2], 7, 2))

    def test_fpr_latency_impure_exact_detection(self):
        self.assertEqual((0, 0, True), ucdd_eval.fpr_and_latency_when_averaging([2, 3, 6], 7, 2))

    def test_fpr_latency_one_false_alarm(self):
        self.assertEqual((0.5, 1, False), ucdd_eval.fpr_and_latency_when_averaging([0], 7, 2))

    def test_fpr_latency_other_false_alarm(self):
        self.assertEqual((0.5, 1, False), ucdd_eval.fpr_and_latency_when_averaging([1], 7, 2))

    def test_fpr_latency_all_false_alarms(self):
        self.assertEqual((1, 1, False), ucdd_eval.fpr_and_latency_when_averaging([0, 1], 7, 2))

    def test_fpr_latency_false_alarm_and_exact(self):
        self.assertEqual((0.5, 0, True), ucdd_eval.fpr_and_latency_when_averaging([0, 2], 7, 2))

    def test_fpr_latency_all_batches_signaled(self):
        self.assertEqual((1, 0, True), ucdd_eval.fpr_and_latency_when_averaging([0, 1, 2, 3, 4, 5, 6], 7, 2))

    def test_fpr_latency_false_alarm_and_late(self):
        self.assertEqual((0.5, 0.5, True), ucdd_eval.fpr_and_latency_when_averaging([0, 4, 5], 7, 2))

    def test_fpr_latency_only_late(self):
        self.assertEqual((0, 0.5, True), ucdd_eval.fpr_and_latency_when_averaging([4, 5], 7, 2))


if __name__ == '__main__':
    unittest.main()
