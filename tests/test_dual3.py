"""Small SDP and exact-replay checks, without experiment artifact dependencies."""
from fractions import Fraction
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qstarc.dual3 import _basis_blocks, _characters, solve_dual3
from qstarc.certification import certify_dual3, replay_dual3_certificate


SDP_AVAILABLE = all(find_spec(name) is not None for name in ('cvxpy', 'clarabel', 'scipy'))
REQUIRES_SDP = unittest.skipUnless(SDP_AVAILABLE, 'requires the optional SDP dependencies')


class Dual3Tests(unittest.TestCase):
    def setUp(self):
        self.rho = np.eye(2)/2
        self.D = np.array([[[0, .5], [.5, 0]]], complex)
        self.B = np.ones((1, 1))

    @REQUIRES_SDP
    def test_full_and_split_certified_qubit(self):
        """The one-parameter qubit has exact attainable inverse-Fisher cost 1."""
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for split in (False, True):
                path = Path(directory)/f'proposal_{split}.npz'
                proposal = solve_dual3(self.rho, self.D, self.B,
                    sign_symmetry=split, output_path=path, tolerance=1e-10)
                certificate = certify_dual3(path)
                lower = Fraction(certificate['risk_lower_exact'])
                self.assertLessEqual(lower, 1)
                self.assertGreater(float(lower), .9999)
                self.assertTrue(certificate['exact_density_normalization'])
                self.assertFalse(certificate['ideal_model_transfer_included'])
                self.assertFalse(certificate['same_finite_SOS_cone_feasibility_claimed'])
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(data['metadata_json'].item())
                self.assertNotIn('solution_path', metadata)
                # A verifier must not trust the candidate eigenvalues/objective.
                with patch('numpy.linalg.eigh', side_effect=RuntimeError('eigh forbidden')), \
                     patch('numpy.linalg.eigvalsh', side_effect=RuntimeError('eigvalsh forbidden')), \
                     patch('numpy.linalg.solve', side_effect=RuntimeError('solve forbidden')), \
                     patch('numpy.linalg.inv', side_effect=RuntimeError('inv forbidden')):
                    replay = replay_dual3_certificate(certificate['certificate_path'])
                self.assertTrue(replay['verified'])
                self.assertEqual(replay['risk_lower_exact'], certificate['risk_lower_exact'])
                results.append(proposal)
            self.assertEqual(results[0]['gram_sizes'], [6])
            self.assertEqual(results[1]['gram_sizes'], [3, 3])
            self.assertLess(abs(results[0]['raw_risk_lower']-results[1]['raw_risk_lower']), 1e-6)

    @REQUIRES_SDP
    def test_support_kernel(self):
        """Pure-state tangent with nonzero support/kernel cross block."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'kernel.npz'
            result = solve_dual3(np.diag([1., 0.]), self.D, self.B,
                output_path=path, sign_symmetry=True, tolerance=1e-10)
            self.assertEqual(sum(result['gram_sizes']), 5)
            certificate = certify_dual3(path)
            self.assertEqual(certificate['rank'], 1)
            self.assertLessEqual(Fraction(certificate['risk_lower_exact']), 1)
            self.assertGreater(certificate['risk_lower'], .9999)
            self.assertTrue(replay_dual3_certificate(certificate['certificate_path'])['verified'])

    @REQUIRES_SDP
    def test_non_normalized_scope_and_tampered_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'unnormalized.npz'
            solve_dual3(self.rho*.75, self.D, self.B, output_path=path)
            certificate = certify_dual3(path)
            self.assertFalse(certificate['exact_density_normalization'])
            self.assertEqual(certificate['trace_rho_exact'], '3/4')
            cert_path = Path(certificate['certificate_path'])
            saved = json.loads(cert_path.read_text())
            saved['risk_lower_exact'] = '1000'
            cert_path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'recomputation mismatch'):
                replay_dual3_certificate(cert_path)

    @REQUIRES_SDP
    def test_rejects_invalid_models_and_symmetry(self):
        for rho in (np.array([[.5, .1], [.1, .5]]), np.diag([0., 1.]), np.diag([1., -.1])):
            with self.assertRaises(ValueError):
                solve_dual3(rho, self.D, self.B)
        bad = self.D.copy()
        bad[0, 0, 1] += .01
        with self.assertRaisesRegex(ValueError, 'exactly Hermitian'):
            solve_dual3(self.rho, bad, self.B)
        with self.assertRaisesRegex(ValueError, 'kernel-kernel'):
            solve_dual3(np.diag([1., 0.]), np.array([np.diag([0., 1.])]), self.B)
        mixed = self.D+np.array([np.diag([1., -1.])])
        with self.assertRaisesRegex(ValueError, 'sign symmetry'):
            solve_dual3(self.rho, mixed, self.B, sign_symmetry=True)
        with self.assertRaises(ValueError):
            solve_dual3(self.rho, self.D, self.B, level=True)

    @REQUIRES_SDP
    def test_basis_tampering_and_optimized_python_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'proposal.npz'
            solve_dual3(self.rho, self.D, self.B, output_path=path)
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            arrays['basis_0'][0] = arrays['basis_0'][1]
            np.savez_compressed(path, **arrays)
            code = (
                'from qstarc.certification import certify_dual3\n'
                'import sys\n'
                'try:\n'
                '    certify_dual3(sys.argv[1])\n'
                'except ValueError:\n'
                '    print("rejected")\n'
                'else:\n'
                '    raise RuntimeError("unsafe acceptance under python -O")\n'
            )
            process = subprocess.run([sys.executable, '-O', '-c', code, str(path)],
                env=os.environ.copy(), capture_output=True, text=True, check=True)
            self.assertIn('rejected', process.stdout)

    def test_qutrit_general_sign_partition_without_large_solve(self):
        rho = np.diag([.5, .25, .25])
        derivatives = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            x = np.zeros((3, 3), complex)
            x[i, j] = x[j, i] = 1
            y = np.zeros((3, 3), complex)
            y[i, j], y[j, i] = -1j, 1j
            derivatives.extend((x, y))
        derivatives.extend((np.diag([0, 1, -1]), np.diag([2, -1, -1])))
        states, scores = _characters(rho, np.asarray(derivatives), np.eye(8), True, None, None)
        self.assertEqual(states, (0, 1, 2))
        self.assertEqual(scores, (1, 1, 2, 2, 3, 3, 0, 0))
        blocks = _basis_blocks(8, 3, 3, 1, states, scores)
        self.assertEqual([len(b) for b in blocks], [35, 35, 35, 30])


if __name__ == '__main__':
    unittest.main()
