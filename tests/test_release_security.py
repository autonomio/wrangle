"""Release integrity and workflow privilege boundaries protect the shipped package."""
import hashlib
from pathlib import Path
import re
import runpy

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
check_release = runpy.run_path(str(ROOT / 'scripts/check_release.py'))['check_release']


@pytest.fixture
def artifacts(tmp_path):
    directory = tmp_path / 'dist'
    directory.mkdir()
    (directory / 'wrangle-1.0.0-py3-none-any.whl').write_bytes(b'checked wheel fixture')
    (directory / 'wrangle-1.0.0.tar.gz').write_bytes(b'checked source fixture')
    return directory


def test_release_hashes_exact_wheel_and_source_archive(artifacts):
    manifest = check_release(artifacts, version='1.0.0', tag='v1.0.0', require_tag=True)
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert digest == hashlib.sha256((artifacts / name).read_bytes()).hexdigest()
    assert len(manifest.read_text().splitlines()) == 2
    assert check_release(artifacts, version='1.0.0', verify=True) == manifest


@pytest.mark.parametrize('tag', [None, 'v0.9.0', '--help', 'v1.0.0\nother'])
def test_release_requires_exact_authorized_version_tag(artifacts, tag):
    with pytest.raises(ValueError):
        check_release(artifacts, version='1.0.0', tag=tag, require_tag=True)
    assert not (artifacts / 'SHA256SUMS').exists()


@pytest.mark.parametrize('mutation', ['wheel', 'source', 'manifest', 'delete'])
def test_release_verification_detects_tampered_distribution(artifacts, mutation):
    check_release(artifacts, version='1.0.0')
    if mutation == 'wheel': (artifacts/'wrangle-1.0.0-py3-none-any.whl').write_bytes(b'tampered')
    if mutation == 'source': (artifacts/'wrangle-1.0.0.tar.gz').write_bytes(b'tampered')
    if mutation == 'manifest': (artifacts/'SHA256SUMS').write_text('untrusted manifest')
    if mutation == 'delete': (artifacts/'SHA256SUMS').unlink()
    with pytest.raises(ValueError):
        check_release(artifacts, version='1.0.0', verify=True)


def test_release_rejects_stale_or_unexpected_artifacts(artifacts):
    (artifacts/'wrangle-0.9.0.tar.gz').write_bytes(b'old')
    with pytest.raises(ValueError):
        check_release(artifacts, version='1.0.0')
    (artifacts/'wrangle-0.9.0.tar.gz').unlink()
    (artifacts/'unreviewed.txt').write_text('unexpected output')
    with pytest.raises(ValueError):
        check_release(artifacts, version='1.0.0')


def test_release_rejects_symlinked_distribution_and_manifest(artifacts, tmp_path):
    target = tmp_path/'outside'; target.write_bytes(b'outside')
    wheel = artifacts/'wrangle-1.0.0-py3-none-any.whl'
    wheel.unlink()
    try:
        wheel.symlink_to(target)
    except OSError:
        pytest.skip('Platform does not permit ordinary symlinks')
    with pytest.raises(ValueError):
        check_release(artifacts, version='1.0.0')
    wheel.unlink(); wheel.write_bytes(b'restored')
    (artifacts/'SHA256SUMS').symlink_to(target)
    with pytest.raises(ValueError):
        check_release(artifacts, version='1.0.0')


def _workflow(name):
    return YAML(typ='safe', pure=True).load((ROOT/'.github/workflows'/name).read_text())


def test_all_workflow_actions_are_pinned_and_checkout_credentials_are_ephemeral():
    for path in (ROOT/'.github/workflows').glob('*.yml'):
        workflow = _workflow(path.name)
        assert not any(value == 'write' for value in workflow.get('permissions', {}).values())
        for job in workflow['jobs'].values():
            assert 0 < job['timeout-minutes'] <= 30
            for step in job['steps']:
                if 'uses' in step:
                    assert re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)?@[a-f0-9]{40}', step['uses'])
                    if step['uses'].startswith('actions/checkout@'):
                        assert step['with']['persist-credentials'] is False


def test_release_privilege_is_separate_from_running_package_code():
    workflow = _workflow('ci-deploy.yml')
    assert workflow['on'] == {'release': {'types': ['published']}}
    build = workflow['jobs']['build']
    assert build.get('permissions', workflow['permissions']) == {'contents': 'read'}
    publish = workflow['jobs']['attest-and-attach']
    assert publish['needs'] == 'build'
    assert publish['permissions'] == {'contents': 'write', 'attestations': 'write', 'id-token': 'write'}
    uses = [step.get('uses', '') for step in publish['steps']]
    assert not any(value.startswith('actions/checkout@') for value in uses)
    operations = '\n'.join(step.get('run', '') for step in publish['steps'])
    assert 'twine upload' not in operations and '--clobber' not in operations
    assert 'gh attestation verify' in operations
    for policy in ('--repo', '--signer-workflow', '--signer-digest', '--source-digest', '--source-ref', '--predicate-type https://slsa.dev/provenance/v1', '--cert-oidc-issuer', '--deny-self-hosted-runners'):
        assert policy in operations
    assert 'dist/*.whl dist/*.tar.gz dist/SHA256SUMS' in operations
    assert 'PYPI_API_TOKEN' not in (ROOT/'.github/workflows/ci-deploy.yml').read_text()


def test_scorecard_publication_obeys_official_restrictions():
    workflow = _workflow('scorecard.yml')
    assert workflow['on']['push']['branches'] == ['master']
    assert 'env' not in workflow and 'defaults' not in workflow
    job = workflow['jobs']['analysis']
    assert 'env' not in job and 'defaults' not in job
    assert job['runs-on'] == 'ubuntu-latest' and job['permissions']['id-token'] == 'write'
    allowed = {'actions/checkout', 'actions/upload-artifact', 'github/codeql-action/upload-sarif', 'ossf/scorecard-action', 'step-security/harden-runner'}
    for step in job['steps']:
        assert 'run' not in step and step['uses'].split('@')[0] in allowed
    assert next(step for step in job['steps'] if step['uses'].startswith('ossf/scorecard-action@'))['with']['publish_results'] is True
