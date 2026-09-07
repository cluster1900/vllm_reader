import unittest

from examples.ch10_benchmark_reasoning import (
    ExperimentRun,
    RequestTrace,
    RuntimeSignals,
    ServiceLevelObjectives,
    compare_runs,
    diagnose,
    pareto_frontier,
    percentile,
    summarize,
)


class RequestTraceTests(unittest.TestCase):
    def test_metric_boundaries(self):
        trace = RequestTrace(1.0, 1.2, (1.5, 1.7, 2.0), 10)
        self.assertAlmostEqual(trace.client_queue_time, 0.2)
        self.assertAlmostEqual(trace.ttft, 0.3)
        self.assertAlmostEqual(trace.itls[0], 0.2)
        self.assertAlmostEqual(trace.itls[1], 0.3)
        self.assertAlmostEqual(trace.e2el, 0.8)
        self.assertAlmostEqual(trace.tpot, 0.25)

    def test_one_token_tpot_is_zero(self):
        trace = RequestTrace(0.0, 0.0, (0.2,), 4)
        self.assertEqual(trace.tpot, 0.0)
        self.assertEqual(trace.itls, ())

    def test_failed_trace_has_no_output_metrics(self):
        trace = RequestTrace(0.0, 0.1, (), 4, success=False, error="timeout")
        self.assertEqual(trace.output_tokens, 0)
        self.assertEqual(trace.ttft, 0.0)
        self.assertEqual(trace.e2el, 0.0)

    def test_rejects_time_travel(self):
        with self.assertRaises(ValueError):
            RequestTrace(1.0, 0.9, (1.1,), 1)
        with self.assertRaises(ValueError):
            RequestTrace(0.0, 0.0, (0.2, 0.1), 1)

    def test_success_requires_token(self):
        with self.assertRaises(ValueError):
            RequestTrace(0.0, 0.0, (), 1)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.traces = [
            RequestTrace(0.0, 0.0, (0.1, 0.2, 0.3), 10),
            RequestTrace(0.0, 0.1, (0.3, 0.5), 20),
            RequestTrace(0.0, 0.2, (), 30, success=False),
        ]

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 2, 3], 50), 2)
        self.assertAlmostEqual(percentile([0, 10], 25), 2.5)

    def test_percentile_empty_is_zero(self):
        self.assertEqual(percentile([], 99), 0.0)

    def test_summarize_counts_only_successful_tokens(self):
        result = summarize(self.traces, duration_s=1.0)
        self.assertEqual(result.completed, 2)
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.total_input_tokens, 30)
        self.assertEqual(result.total_output_tokens, 5)
        self.assertEqual(result.request_throughput, 2.0)
        self.assertEqual(result.output_throughput, 5.0)
        self.assertEqual(result.total_token_throughput, 35.0)

    def test_summarize_flattens_itl_samples(self):
        result = summarize(self.traces, duration_s=1.0)
        self.assertAlmostEqual(result.mean_itl_s, (0.1 + 0.1 + 0.2) / 3)

    def test_goodput_requires_all_slos(self):
        slos = ServiceLevelObjectives(ttft_s=0.15, tpot_s=0.15)
        result = summarize(self.traces, duration_s=1.0, slos=slos)
        self.assertEqual(result.request_goodput, 1.0)

    def test_without_slo_goodput_is_zero(self):
        self.assertEqual(summarize(self.traces, 1.0).request_goodput, 0.0)

    def test_rejects_zero_duration(self):
        with self.assertRaises(ValueError):
            summarize(self.traces, 0.0)


class ExperimentTests(unittest.TestCase):
    def test_compare_runs(self):
        baseline = ExperimentRun("base", (100, 100), (0.2, 0.2))
        treatment = ExperimentRun("new", (110, 110), (0.22, 0.22))
        result = compare_runs(baseline, treatment)
        self.assertAlmostEqual(result.throughput_change_pct, 10)
        self.assertAlmostEqual(result.p99_latency_change_pct, 10)

    def test_correctness_gate_uses_error_rate(self):
        baseline = ExperimentRun("base", (100,), (0.2,))
        treatment = ExperimentRun("new", (120,), (0.2,), (0.01, 0.03))
        self.assertFalse(compare_runs(baseline, treatment, 0.01).passes_correctness_gate)
        self.assertTrue(compare_runs(baseline, treatment, 0.02).passes_correctness_gate)

    def test_cv_exposes_noisy_measurements(self):
        run = ExperimentRun("noisy", (80, 100, 120), (0.2,))
        self.assertGreater(run.throughput_cv, 0.1)

    def test_pareto_frontier(self):
        fast = ExperimentRun("fast", (120,), (0.3,))
        low_latency = ExperimentRun("low", (90,), (0.1,))
        dominated = ExperimentRun("bad", (80,), (0.4,))
        names = [run.name for run in pareto_frontier([fast, low_latency, dominated])]
        self.assertEqual(names, ["low", "fast"])

    def test_identical_tradeoff_points_remain(self):
        a = ExperimentRun("a", (100,), (0.2,))
        b = ExperimentRun("b", (100,), (0.2,))
        self.assertEqual(len(pareto_frontier([a, b])), 2)


class DiagnosisTests(unittest.TestCase):
    def test_kv_pressure(self):
        signals = RuntimeSignals(3, 0.95, 2, 0.8, 0.1, 0.1)
        self.assertIn("kv_capacity_pressure", diagnose(signals))

    def test_gpu_saturation(self):
        signals = RuntimeSignals(3, 0.7, 0, 0.95, 0.1, 0.0)
        self.assertIn("gpu_saturated", diagnose(signals))

    def test_communication_and_cpu_hypotheses(self):
        signals = RuntimeSignals(1, 0.5, 0, 0.6, 0.4, 0.3)
        result = diagnose(signals)
        self.assertIn("communication_heavy", result)
        self.assertIn("cpu_or_control_plane_gap", result)

    def test_balanced_signals_are_inconclusive(self):
        signals = RuntimeSignals(0, 0.5, 0, 0.8, 0.1, 0.1)
        self.assertEqual(diagnose(signals), ("insufficient_or_balanced_signals",))


if __name__ == "__main__":
    unittest.main()
