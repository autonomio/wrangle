# Security requirements

Wrangle is a local preparation library and CLI. It accepts researcher-selected
files and explicit scientific protocols; it does not provide a network service,
user account system, encryption service or execution sandbox.

## Required behavior

- Recipe files contain one strict YAML mapping with ordinary finite scalar values,
  lists and mappings. Reject tags, aliases, anchors, duplicate keys, recursive or
  overly deep values, ambiguous numeric identifiers and executable objects.
- Resolve only registered operations and allowlisted expressions, types and
  options. Treat source text, column names and descriptions as data. Reject Python
  callbacks/UDF plans; never evaluate text as Python, shell commands or templates.
- Keep required guided decisions unresolved until supplied. Check recorded choices
  against controlled operations; drafting is not data validation.
- Enforce native keys, cardinality, ordering, precision, missingness, units,
  exclusions and observation transitions. A failed contract prevents publication.
- Capture sources, retain input/protocol/output identity, and verify parent data,
  recipe, report and evidence before reuse. Reject changed or inconsistent bundles.
- Publish to a new directory atomically with exclusive destination control.
  Concurrent writers or unsupported publication mechanisms must fail explicitly.
- Escape untrusted text in notebook HTML and terminal presentation. Structured
  output preserves literal values as data, without prompting or guessing.
- Audit dependencies and analyze implementation before release. Document findings,
  resolved versions and artifact verification instead of claiming a badge from policy alone.

## What researchers must decide and protect

A recipe's scientific correctness depends on the supplied observation unit,
identifiers, physical units, missing codes, exclusions, statistical choices and
analysis requirements. Wrangle cannot establish those from observed values.
Tests and receipts establish the declared behavior, not scientific truth.

Receipts include source paths, examples, identifiers and exclusion evidence;
they may contain sensitive study information. Store the output with the same
access controls as its input. Wrangle does not anonymize or encrypt these files
and does not submit them to a network service.

Use files, Python objects and environments you are authorized to access. Native
Python access can execute ordinary Python outside Wrangle; a recipe restriction
is not a sandbox around its caller. A compromised interpreter, dependency,
operating system or privileged local process is outside the package's guarantees.

SHA-256 records snapshot consistency, not the identity or authorization of a
source's author. A coordinated replacement of data and all evidence can create a
new consistent bundle. Release-artifact signatures address software origin through
a separate identity/transparency mechanism; see [release verification](security/releases.md).

Large or adversarial files can exhaust memory, disk or time. Native decoders can
have vulnerabilities; dependency monitoring reduces but does not eliminate that
risk. Disk preparation preserves contracts but gives no fixed peak-memory bound.
Use operating-system resource limits and isolated accounts when processing
untrusted datasets. Do not treat a saved path as a filesystem permission boundary.

See [architecture](architecture.md), [assurance evidence](security/assurance.md),
and the [security policy](project/SECURITY.md) for reports, supported versions
and response handling. Hash/replay equivalence across different Polars releases,
hardware, execution engines or reordered inputs is not promised.
