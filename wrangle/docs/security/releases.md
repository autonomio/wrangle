# Verify a release

New native-Polars releases are built by `.github/workflows/ci-deploy.yml` from a
published version tag. Its build job tests the source, validates the generated
manual, installs and exercises the actual wheel, scans dependencies and checks
package metadata. `scripts/check_release.py --require-tag` requires that the tag,
wheel and source archive all match the version in `pyproject.toml`.

A separate job signs provenance for the wheel, source archive and `SHA256SUMS`
using GitHub Actions OIDC and Sigstore through `actions/attest`. It verifies the
repository, workflow, source commit, tag and hosted-runner identity before
attaching artifacts, `attestation.json` and `provenance.intoto.jsonl` to the release.
The JSONL file contains the unchanged signed DSSE envelope from the verified
Sigstore bundle, making the existing SLSA provenance available in the standard
in-toto distribution format. Keep the complete bundle for certificate and trust
verification. Assets are never silently replaced. A checksum establishes byte consistency; signed provenance
establishes the release workflow identity. Neither establishes scientific validity.

## Publication status

Check the [GitHub release](https://github.com/autonomio/wrangle/releases) and its
completed artifact workflow. A release entry or workflow definition alone does
not establish that signed assets exist: require the wheel, source archive,
checksums and attestation bundle, then perform the verification below. Legacy
0.x releases lack this provenance. OpenSSF attainment is assessed separately on
the [badge page](https://www.bestpractices.dev/en/projects/15135).

## Download and verify

Use the official GitHub CLI with `gh attestation verify` support. Select a release
and copy its full commit SHA from the repository, independently of the downloaded
artifacts. Replace the two values below with that tag and commit:

```sh
release_tag=v1.0.0
release_commit=REPLACE_WITH_FULL_RELEASE_COMMIT_SHA
mkdir wrangle-release-verification
cd wrangle-release-verification
gh release download "$release_tag" --repo autonomio/wrangle
sha256sum --check SHA256SUMS
for artifact in *.whl *.tar.gz SHA256SUMS; do
  gh attestation verify "$artifact" --bundle attestation.json \
    --repo autonomio/wrangle \
    --signer-workflow autonomio/wrangle/.github/workflows/ci-deploy.yml \
    --signer-digest "$release_commit" --source-digest "$release_commit" \
    --source-ref "refs/tags/$release_tag" \
    --predicate-type https://slsa.dev/provenance/v1 \
    --cert-oidc-issuer https://token.actions.githubusercontent.com \
    --deny-self-hosted-runners
done
```

On macOS, `shasum -a 256 --check SHA256SUMS` replaces `sha256sum --check`.
Verification requires Internet access to retrieve trusted verification material.
Stop on a missing artifact, failed checksum, failed signature or identity mismatch;
report the release URL and failing command privately if tampering is suspected.
Install the verified wheel only after all checks succeed.

## Maintainers

Use the required contributor checks before publication.
`python scripts/check_reproducible_build.py` builds twice with a fixed timestamp
and hash seed and compares both distributions byte-for-byte in one environment.
Different operating systems or tool versions may produce different bytes. Build from a clean commit;
never add credentials or participant data to a release. Version tags are immutable.
Create release notes describing behavior, compatibility and any disclosed security
fixes. The artifact workflow uses short-lived GitHub credentials with separate
read-only build and restricted signing/publication jobs. PyPI publication is not
configured by this workflow; any future trusted publisher must be verified against
the intended repository, workflow and environment before enabling it.
