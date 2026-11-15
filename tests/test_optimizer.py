"""Portable numerical regressions; no external experiment workspace is used."""
from pathlib import Path
import json
import unittest
import numpy as np

try:
    import jax
    from qstarc import optimizer
except ModuleNotFoundError as error:
    if error.name not in ("jax", "jaxlib"):
        raise
    optimizer = None


def complex_array(record):
    return np.asarray(record["real"]) + 1j * np.asarray(record["imag"])


@unittest.skipIf(optimizer is None, "JAX is required for optimizer tests")
class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads((Path(__file__).parent / "data/frozen_reference.json").read_text())
        cls.rho = complex_array(cls.reference["rho"])
        cls.derivatives = complex_array(cls.reference["derivatives"])
        cls.Q0 = complex_array(cls.reference["initial_Q"])

    def test_fixed_frame_fisher_and_metric_gradient(self):
        self.assertEqual(self.reference["source_sha256"],
                         "62293ff0ea05ceda91113bd0e22197b5dcaf2fa6a94d9a66581c8508921f901e")
        fisher = np.asarray(optimizer.classical_fisher_stiefel(self.Q0, self.rho, self.derivatives))
        value, gradient = optimizer.value_and_rgrad_Q(self.Q0, self.rho, self.derivatives)
        np.testing.assert_allclose(fisher, self.reference["initial_fisher"], rtol=1e-11, atol=1e-12)
        self.assertAlmostEqual(float(value), self.reference["initial_value"], delta=1e-10)
        np.testing.assert_allclose(gradient, complex_array(self.reference["initial_rgrad"]), rtol=1e-10, atol=1e-11)
        tangent_residual = self.Q0.conj().T @ gradient + gradient.conj().T @ self.Q0
        np.testing.assert_allclose(tangent_residual, 0, rtol=0, atol=1e-11)

    def test_projected_adam_last_iterate_regression(self):
        settings = self.reference["settings"]
        q = optimizer.stiefel_adam(self.rho, self.derivatives, settings["mout"],
            seed=settings["seed"], adam_steps=settings["adam_steps"], adam_lr=settings["adam_lr"],
            adam_chunk=settings["adam_chunk"], verbose=False)
        value = float(optimizer.value_Q(q, self.rho, self.derivatives))
        fisher = np.asarray(optimizer.classical_fisher_stiefel(q, self.rho, self.derivatives))
        self.assertAlmostEqual(value, self.reference["adam_last_value"], delta=1e-10)
        np.testing.assert_allclose(fisher, self.reference["adam_last_fisher"], rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(np.asarray(q).conj().T @ q, np.eye(2), rtol=0, atol=1e-12)

    def test_short_rcg_fixed_start_regression(self):
        result = optimizer.stiefel_rcg(self.rho, self.derivatives, self.Q0, maxiter=5, verbose=False)
        reference = self.reference["rcg_from_initial"]
        self.assertAlmostEqual(result["minimum"], reference["minimum"], delta=1e-10)
        np.testing.assert_allclose(result["fisher"], reference["fisher"], rtol=1e-10, atol=1e-11)
        self.assertEqual(result["status"], reference["status"])
        self.assertEqual(result["iterations"], reference["iterations"])
        self.assertLess(result["completeness_error"], 1e-12)

    def test_legacy_combined_call_regression(self):
        result = optimizer.minimizeQ_stiefel(self.rho, self.derivatives, **self.reference["settings"])
        reference = self.reference["combined"]
        self.assertAlmostEqual(result["minimum"], reference["minimum"], delta=1e-10)
        np.testing.assert_allclose(result["fisher"], reference["fisher"], rtol=1e-10, atol=1e-11)
        self.assertEqual(result["status"], reference["status"])
        self.assertEqual(result["converged"], reference["converged"])
        self.assertEqual(result["line_search_evaluations"], reference["line_search_evaluations"])
        self.assertEqual(result["Q"].dtype, np.complex128)
        self.assertEqual(result["fisher"].dtype, np.float64)
        self.assertLess(result["completeness_error"], 1e-12)

    def test_fisher_uses_tiny_probabilities_without_a_floor(self):
        # Fisher's homogeneous scaling law is valid before imposing completeness.
        scale = 1e-12
        scaled = np.asarray(optimizer.classical_fisher_stiefel(scale*self.Q0, self.rho, self.derivatives))
        np.testing.assert_allclose(scaled/(scale*scale), self.reference["initial_fisher"], rtol=1e-11, atol=1e-12)
        padded = np.concatenate([self.Q0, np.zeros((1, 2), complex)])
        fisher = np.asarray(optimizer.classical_fisher_stiefel(padded, self.rho, self.derivatives))
        np.testing.assert_allclose(fisher, self.reference["initial_fisher"], rtol=1e-11, atol=1e-12)

    def test_singular_fisher_is_not_regularized(self):
        q = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], complex)/np.sqrt(2)
        value = float(optimizer.value_Q(q, self.rho, self.derivatives))
        self.assertTrue(np.isinf(value))

    def test_weighted_wrapper_coordinates_and_dual_handoff(self):
        weight = np.array([[2., .2, .1], [.2, 1.5, -.15], [.1, -.15, .9]])
        result = optimizer.optimize_povm(self.rho, self.derivatives, 6, seed=7,
            weight=weight, adam_steps=0, rcg_steps=0, verbose=False)
        original_fisher = np.asarray(optimizer.classical_fisher_stiefel(result["Q"], self.rho, self.derivatives))
        expected_risk = float(np.trace(weight @ np.linalg.inv(original_fisher)))
        self.assertAlmostEqual(result["risk"], expected_risk, delta=1e-10)
        np.testing.assert_allclose(result["fisher_original"], original_fisher, rtol=1e-12, atol=1e-12)
        white_fisher = np.asarray(optimizer.classical_fisher_stiefel(result["Q"], result["rho"], result["D"]))
        np.testing.assert_allclose(result["fisher"], white_fisher, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.array_equal(result["B"], result["B"].T))
        np.testing.assert_allclose(result["B"] @ white_fisher, np.eye(3), rtol=0, atol=1e-12)
        self.assertFalse(result["risk_is_certified_upper"])
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "maxiter")
        self.assertEqual(result["measurement_outcomes"], 6)

    def test_multistart_selects_finite_final_risk_preserving_status(self):
        result = optimizer.optimize_multistart(self.rho, self.derivatives, 6,
            seeds=(1, 7), adam_steps=0, rcg_steps=0, verbose=False)
        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual(result["best"]["risk"], min(run["risk"] for run in result["runs"]))
        self.assertIs(result["best"], result["runs"][result["best_index"]])
        self.assertTrue(all(not run["converged"] for run in result["runs"]))
        with self.assertRaises(FloatingPointError):
            optimizer.optimize_multistart(self.rho, np.zeros_like(self.derivatives), 6,
                seeds=(1, 7), adam_steps=0, rcg_steps=0, verbose=False)

    def test_invalid_inputs_are_rejected(self):
        options = dict(adam_steps=0, rcg_steps=0, verbose=False)
        bad_state = self.rho.copy(); bad_state[0, 0] += 1e-6
        with self.assertRaises(ValueError):
            optimizer.minimizeQ_stiefel(bad_state, self.derivatives, 6, **options)
        with self.assertRaises(ValueError):
            optimizer.minimizeQ_stiefel(self.rho, self.derivatives, 6, seed=True, **options)
        with self.assertRaises(ValueError):
            optimizer.minimizeQ_stiefel(self.rho, self.derivatives, True, **options)
        with self.assertRaises(ValueError):
            optimizer.minimizeQ_stiefel(self.rho, self.derivatives, 3, **options)
        with self.assertRaises(ValueError):
            optimizer.optimize_multistart(self.rho, self.derivatives, 6, seeds=(), **options)
        result = optimizer.minimizeQ_stiefel(self.rho, self.derivatives, np.int64(6), seed=np.int64(7), **options)
        self.assertTrue(np.isfinite(result["minimum"]))


if __name__ == "__main__":
    unittest.main()
