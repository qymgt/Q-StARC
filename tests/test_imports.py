"""Check that importing the package does not activate optional backends."""

import subprocess
import sys
import unittest


class ImportTests(unittest.TestCase):
    def test_base_import_is_lazy(self):
        code = (
            "import sys, qstarc; "
            "assert qstarc.__version__ == '1.0.0'; "
            "assert not ({'jax', 'cvxpy', 'clarabel'} & sys.modules.keys())"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_unknown_attribute(self):
        import qstarc

        with self.assertRaises(AttributeError):
            getattr(qstarc, "not_a_qstarc_function")


if __name__ == "__main__":
    unittest.main()
