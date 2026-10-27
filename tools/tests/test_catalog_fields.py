"""check_catalog rejects a skills.yaml key nothing reads and an unknown intel-hw-class."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import validate_skills as v  # noqa: E402


def errors_for(entry: dict[str, str]) -> list[str]:
    report = v.Report()
    entry = {"name": "dpnp-quickstart", "maintainer": "someone", **entry}
    v.check_catalog({"dpnp-quickstart": entry}, {"dpnp-quickstart": "tool-skill"}, report)
    return report.errors


class CatalogFields(unittest.TestCase):
    def test_known_fields_pass(self) -> None:
        self.assertEqual(errors_for({"intel-hw-class": "gpu", "intel-products": "dpnp"}), [])

    def test_empty_hw_class_means_not_applicable(self) -> None:
        self.assertEqual(errors_for({"intel-hw-class": ""}), [])

    def test_misspelt_key_is_an_error(self) -> None:
        errors = errors_for({"intel-hw-clas": "gpu"})
        self.assertEqual(len(errors), 1)
        self.assertIn("unknown key 'intel-hw-clas'", errors[0])

    def test_unknown_hw_class_is_an_error(self) -> None:
        errors = errors_for({"intel-hw-class": "GPU"})
        self.assertEqual(len(errors), 1)
        self.assertIn("intel-hw-class must be one of", errors[0])


if __name__ == "__main__":
    unittest.main()
