# Adversarial protocol fuzzing

Two Atheris targets exercise untrusted scientific protocols without opening
research files, evaluating source text, calling external programs or networking:

- `scripts/fuzz_protocol.py`: strict YAML parsing, finite JSON values, exact
  parse/dump round trips and deterministic serialization.
- `scripts/fuzz_expression.py`: mutated native expression trees through public
  `wrangle.prepare` on three synthetic rows; source immutability, preserved
  observation keys and repeatable data/receipts.

These are checkout paths. Seed corpora live in `scripts/fuzz_corpus/` and include
valid expressions, scientific identifier/scalar cases, aliases, duplicate keys,
executable YAML tags, casts, regex replacements and arithmetic overflow.
Rejected inputs may raise only the documented `WrangleError` boundary; unexpected
exceptions, native panics and failed properties stop the run and retain input.

`.github/workflows/fuzzing.yml` runs both targets for 60 seconds each on relevant
pull requests and master changes, and for five minutes each every week. Jobs use
one Polars thread, an 8 KiB input limit, a ten-second per-input timeout, a 2 GiB
memory limit, a 15-minute job timeout and cancellation of superseded work. Each
run retains its expanded corpora and failure artifacts for 14 days. Seeds are
committed; generated inputs are evidence rather than trusted recipe suggestions.

## Reproduce and recover

Use an isolated Linux x86_64 environment with Python 3.13:

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements-ci.txt -r requirements-audit.txt -r requirements-fuzz.txt
python -m pip install --no-deps --no-build-isolation -e .
mkdir -p fuzz-results/protocol fuzz-results/expression
cp scripts/fuzz_corpus/protocol/* fuzz-results/protocol/
cp scripts/fuzz_corpus/expression/* fuzz-results/expression/
POLARS_MAX_THREADS=1 python scripts/fuzz_protocol.py fuzz-results/protocol -artifact_prefix=fuzz-results/protocol/ -max_len=8192 -timeout=10 -rss_limit_mb=2048 -max_total_time=60
POLARS_MAX_THREADS=1 python scripts/fuzz_expression.py fuzz-results/expression -artifact_prefix=fuzz-results/expression/ -max_len=8192 -timeout=10 -rss_limit_mb=2048 -max_total_time=60
```

Download a failing workflow's corpus artifact, and pass its crash file as the
sole positional argument to the corresponding target. Fix the exposed defect,
add a focused regression to the ordinary tests, and retain a minimal seed before
rerunning the target. Follow `SECURITY.md` for privately reporting security bugs.
Do not catch additional exception types merely to make fuzzing pass.

Atheris instruments Python branches; the distributed Polars native extension
does not carry sanitizer instrumentation. This checks the Python contracts and
observable native failures, not arbitrary native-memory safety. Coverage-guided
fuzzing is bounded evidence, not proof that every hostile protocol is safe.
OpenSSF Scorecard 5.5 recognizes the actual Atheris imports; its detection alone
does not establish that a particular workflow run succeeded.
