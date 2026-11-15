"""Coordinate and weighting identities, independent of external experiments."""
import importlib.util
import unittest
import numpy as np
from qstarc import models


def sld_information_for_diagonal_state(rho, derivatives):
    rho, derivatives = np.asarray(rho), np.asarray(derivatives)
    eigenvalues = np.diag(rho).real
    denominator = eigenvalues[:, None] + eigenvalues[None, :]
    return np.array([[np.sum(2*a*b.T/denominator).real for b in derivatives] for a in derivatives])


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.derivatives = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)/2

    def test_nondiagonal_weight_whitening_identity(self):
        weight = np.array([[2, .3, -.2], [.3, 1.5, .1], [-.2, .1, .8]])
        original_fisher = np.array([[1.2, .1, -.05], [.1, .8, .12], [-.05, .12, 1.6]])
        transformed, T = models.whiten_derivatives(self.derivatives, weight)
        np.testing.assert_allclose(T.T @ weight @ T, np.eye(3), rtol=0, atol=1e-14)
        np.testing.assert_allclose(transformed, np.einsum("ji,jab->iab", T, self.derivatives), rtol=0, atol=1e-15)
        white_fisher = T.T @ original_fisher @ T
        self.assertAlmostEqual(float(np.trace(np.linalg.inv(white_fisher))),
                               float(np.trace(weight @ np.linalg.inv(original_fisher))), delta=1e-13)
        np.testing.assert_allclose(models.fisher_to_original(white_fisher, T), original_fisher, rtol=1e-13, atol=1e-14)

    def test_weight_is_not_clipped_or_regularized(self):
        for invalid in (np.diag([1, 1, 0]), np.diag([1, 1, -1]),
                        np.array([[1., .2, 0], [0, 1, 0], [0, 0, 1]]),
                        np.full((3, 3), np.nan), np.eye(2)):
            with self.subTest(weight=invalid):
                with self.assertRaises(ValueError):
                    models.whiten_derivatives(self.derivatives, invalid)
        with self.assertRaises(ValueError):
            models.whiten_derivatives(self.derivatives, np.eye(3)+1j*np.eye(3))


@unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is required for legacy qubit construction")
class QubitCoordinateTests(unittest.TestCase):
    def test_sld_and_bloch_information_have_distinct_normalizations(self):
        r = .5
        for ncopy in (1, 2):
            with self.subTest(ncopy=ncopy):
                rho, normalized = models.qubit_model_sld(r, ncopy)
                rho_bloch, bloch = models.qubit_model_bloch(r, ncopy)
                np.testing.assert_array_equal(rho, rho_bloch)
                np.testing.assert_allclose(sld_information_for_diagonal_state(rho, normalized), ncopy*np.eye(3), rtol=1e-13, atol=1e-13)
                expected = ncopy*np.diag([1, 1, 1/(1-r*r)])
                np.testing.assert_allclose(sld_information_for_diagonal_state(rho_bloch, bloch), expected, rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(np.asarray(normalized)[2], np.sqrt(1-r*r)*np.asarray(bloch)[2], rtol=1e-14, atol=1e-14)

    def test_product_rule_and_legacy_alias(self):
        rho1, D1 = map(np.asarray, models.qubit_model_sld(.35, 1))
        rho2, D2 = map(np.asarray, models.qubit_model_sld(.35, 2))
        np.testing.assert_allclose(rho2, np.kron(rho1, rho1), rtol=0, atol=1e-15)
        for i in range(3):
            np.testing.assert_allclose(D2[i], np.kron(D1[i], rho1)+np.kron(rho1, D1[i]), rtol=0, atol=1e-15)
        legacy_rho, legacy_D = models.qbit_model_n(.35, 2)
        np.testing.assert_array_equal(legacy_rho, rho2)
        np.testing.assert_array_equal(legacy_D, D2)

    def test_invalid_regular_coordinate_inputs(self):
        for radius in (1, -1, np.nan, np.inf, .2j):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    models.qubit_model_sld(radius)
                with self.assertRaises(ValueError):
                    models.qubit_model_bloch(radius)
        for copies in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                models.qubit_model_sld(.2, copies)
        rho, D = models.qubit_model_sld(0, np.int64(1))
        self.assertEqual(rho.shape, (2, 2)); self.assertEqual(D.shape, (3, 2, 2))


if __name__ == "__main__":
    unittest.main()
