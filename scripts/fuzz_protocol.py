"""Coverage-guided fuzzing of the strict YAML boundary; see docs/security/fuzzing.md."""
import sys

import atheris

with atheris.instrument_imports():
    from _fuzz_targets import fuzz_protocol


@atheris.instrument_func
def test_one_input(data):
    fuzz_protocol(data)


if __name__ == "__main__":
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
