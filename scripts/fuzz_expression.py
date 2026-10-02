"""Coverage-guided fuzzing of declarative native expressions; never executes input code."""
import sys

import atheris

with atheris.instrument_imports():
    from _fuzz_targets import fuzz_expression


@atheris.instrument_func
def test_one_input(data):
    fuzz_expression(data)


if __name__ == "__main__":
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
