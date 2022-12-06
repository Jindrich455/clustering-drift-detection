import unittest

import ucdd_eval


class MyTestCase(unittest.TestCase):
    def test_fpr_latency_no_drift_detected(self):
        self.assertEqual((0, 1), ucdd_eval.fpr_and_latency_when_averaging([], 7, 2))

    def test_fpr_latency_only_false_alarm(self):
        self.assertEqual((1, 1), ucdd_eval.fpr_and_latency_when_averaging([0], 7, 2))

    def test_fpr_latency_false_alarm_and_late(self):
        self.assertEqual((1, 0.4), ucdd_eval.fpr_and_latency_when_averaging([0, 4, 5], 7, 2))

    def test_fpr_latency_only_late(self):
        self.assertEqual((0, 0.4), ucdd_eval.fpr_and_latency_when_averaging([4, 5], 7, 2))

    def test_fpr_latency_detection_on_time(self):
        self.assertEqual((0, 0), ucdd_eval.fpr_and_latency_when_averaging([2], 7, 2))


if __name__ == '__main__':
    unittest.main()
