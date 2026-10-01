"""Execute the shipped manual, catalog and CLI from the actual built wheel."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ('research_batch.py', 'preparation_workflows.py', 'frozen_parameters.py', 'inspection_workflow.py', 'expression_workflow.py', 'disk_preparation.py', 'human_workflow.py', 'start_workflow.py')
REQUIRED = {'wrangle/AGENTS.md', 'wrangle/docs/README.md', 'wrangle/docs/recipes.md', 'wrangle/docs/migration.md', 'wrangle/docs/operations.json', 'wrangle/_cli.py', 'wrangle/__main__.py', 'wrangle/_protocol.py', 'wrangle/_start.py', 'wrangle/_presentation.py', 'wrangle/docs/getting_started.md', 'wrangle/docs/starter/recipe.yaml', 'wrangle/docs/starter/samples.csv', 'wrangle/docs/starter/metadata.csv', 'wrangle/docs/starter/README.md', *(f'wrangle/docs/{name}' for name in WORKFLOWS)}

REQUIRED.update({'wrangle/docs/architecture.md', 'wrangle/docs/security.md',
                 'wrangle/docs/security/assurance.md', 'wrangle/docs/security/releases.md',
                 'wrangle/docs/security/openssf-evidence.yaml',
                 *(f'wrangle/docs/project/{name}' for name in ('CONTRIBUTING.md', 'SECURITY.md', 'GOVERNANCE.md', 'CODE_OF_CONDUCT.md', 'ROADMAP.md', 'CHANGELOG.md'))})


def run():
    wheels = sorted((ROOT / 'dist').glob('wrangle-*.whl'))
    sdists = sorted((ROOT / 'dist').glob('wrangle-*.tar.gz'))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit('Build exactly one current wheel and sdist before distribution verification.')
    with zipfile.ZipFile(wheels[0]) as wheel:
        missing = REQUIRED - set(wheel.namelist())
        if missing:
            raise SystemExit('Wheel is missing agent files: ' + ', '.join(sorted(missing)))
    with tarfile.open(sdists[0]) as sdist:
        files = {name.split('/', 1)[1] for name in sdist.getnames() if '/' in name}
        needed = (REQUIRED - {'wrangle/AGENTS.md'}) | {'AGENTS.md', 'scripts/build_catalog.py', 'scripts/verify_distribution.py'}
        if needed - files:
            raise SystemExit('Sdist is missing agent files: ' + ', '.join(sorted(needed - files)))
    with tempfile.TemporaryDirectory(prefix='wrangle-wheel-check-') as temporary:
        target = Path(temporary) / 'installed'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--disable-pip-version-check', '--target', str(target), str(wheels[0])], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        environment = {**os.environ, 'PYTHONPATH': str(target)}
        code = '''from pathlib import Path
import json, runpy, sys
import wrangle
from wrangle._catalog import catalog, operation_document
root = Path(wrangle.__file__).parent
assert root.parent == Path(sys.argv[1])
document = catalog()
assert json.loads((root / 'docs/operations.json').read_text()) == document
for entry in document['operations']:
    assert json.loads((root / entry['manual']).read_text()) == operation_document(entry['name'], document)
for name in sys.argv[2:]:
    namespace = runpy.run_path(str(root / 'docs' / name))
    namespace['run']()
print(json.dumps({'installed': str(root), 'operations': len(catalog()['operations']), 'workflows': len(sys.argv[2:])}))
'''
        result = subprocess.run([sys.executable, '-c', code, str(target), *WORKFLOWS], cwd=temporary, env=environment, check=True, text=True, capture_output=True)
        metadata = json.loads(result.stdout)
        entrypoint = next(path for path in target.rglob('wrangle.exe' if os.name == 'nt' else 'wrangle') if path.is_file())
        cli = subprocess.run([str(entrypoint), 'catalog', 'unpivot', '--json'], cwd=temporary, env=environment, check=True, text=True, capture_output=True)
        assert json.loads(cli.stdout)['operations'][0]['name'] == 'unpivot'
        study = Path(temporary) / 'my study'
        starter = subprocess.run([str(entrypoint), 'example', str(study)], cwd=temporary, env=environment, check=True, text=True, capture_output=True)
        assert starter.stdout.startswith('Example ready:') and not starter.stderr
        observed = subprocess.run([str(entrypoint), 'inspect', 'samples.csv'], cwd=study, env=environment, check=True, text=True, capture_output=True)
        assert 'Rows: 3 | Columns: 3' in observed.stdout and '\"001\"' in observed.stdout and not observed.stderr
        arguments = ['prepare', 'recipe.yaml', '--source', 'measurements=samples.csv', '--source', 'metadata=metadata.csv']
        human = subprocess.run([str(entrypoint), *arguments, '--output', 'prepared-human'], cwd=study, env=environment, check=True, text=True, capture_output=True)
        assert 'Rows: 3 -> 2' in human.stdout and 'Excluded observations: 1' in human.stdout and not human.stderr
        machine = subprocess.run([sys.executable, '-m', 'wrangle', *arguments, '--output', 'prepared-agent', '--json'], cwd=study, env=environment, check=True, text=True, capture_output=True)
        receipt = json.loads(machine.stdout)
        assert receipt == json.loads((study / 'prepared-human/receipt.json').read_text(encoding='utf-8'))
        assert receipt['output']['rows'] == 2 and receipt['units'] == {'mass_g': 'g'}
        assert (study / 'prepared-agent/report.txt').read_bytes() == (study / 'prepared-human/report.txt').read_bytes()
        assert (study / 'prepared-agent/recipe.yaml').is_file()
        assert not (study / 'prepared-agent/recipe.json').exists()
        # Exercise installed entrypoints against the user's first-protocol path.
        guide_arguments = ['start', 'samples.csv', '--metadata', 'metadata.csv']
        pending = subprocess.run([str(entrypoint), *guide_arguments, '--output', 'pending-protocol', '--json'], cwd=study, env=environment, check=True, text=True, capture_output=True)
        proposal = json.loads(pending.stdout)
        assert proposal['ready'] is False and proposal['data_validated'] is False and not pending.stderr
        blocked = subprocess.run([sys.executable, '-m', 'wrangle', 'prepare', 'pending-protocol/recipe.yaml', '--source', 'measurements=samples.csv', '--source', 'metadata=metadata.csv', '--output', 'blocked-preparation', '--json'], cwd=study, env=environment, text=True, capture_output=True)
        assert blocked.returncode == 1 and json.loads(blocked.stderr)['code'] == 'UNRESOLVED_PROTOCOL'
        assert not (study / 'blocked-preparation').exists()
        answer_code = "from pathlib import Path; import runpy, wrangle; from wrangle._protocol import dump_recipe; workflow = runpy.run_path(str(Path(wrangle.__file__).parent / 'docs/start_workflow.py')); Path('answers.yaml').write_text(dump_recipe(workflow['supplied_answers']()), encoding='utf-8')"
        subprocess.run([sys.executable, '-c', answer_code], cwd=study, env=environment, check=True, text=True, capture_output=True)
        guided_human = subprocess.run([str(entrypoint), *guide_arguments, '--answers', 'answers.yaml', '--output', 'resolved-protocol'], cwd=study, env=environment, check=True, text=True, capture_output=True)
        assert 'Protocol ready to prepare' in guided_human.stdout and not guided_human.stderr
        guided_agent = subprocess.run([sys.executable, '-m', 'wrangle', *guide_arguments, '--answers', 'answers.yaml', '--json'], cwd=study, env=environment, check=True, text=True, capture_output=True)
        proposal = json.loads(guided_agent.stdout)
        assert proposal['ready'] and proposal['data_validated'] is False and not proposal['questions']
        guided_arguments = ['prepare', 'resolved-protocol/recipe.yaml', '--source', 'measurements=samples.csv', '--source', 'metadata=metadata.csv']
        guided_result = subprocess.run([str(entrypoint), *guided_arguments, '--output', 'guided-human'], cwd=study, env=environment, check=True, text=True, capture_output=True)
        assert 'One specimen per row' in guided_result.stdout and not guided_result.stderr
        guided_result = subprocess.run([sys.executable, '-m', 'wrangle', *guided_arguments, '--output', 'guided-agent', '--json'], cwd=study, env=environment, check=True, text=True, capture_output=True)
        guided_receipt = json.loads(guided_result.stdout)
        assert guided_receipt == json.loads((study / 'guided-human/receipt.json').read_text(encoding='utf-8'))
        assert guided_receipt['output']['rows'] == 2 and guided_receipt['units'] == {'mass_mg': 'mg'}
        assert (study / 'guided-human/report.txt').read_bytes() == (study / 'guided-agent/report.txt').read_bytes()
        invalid = subprocess.run([sys.executable, '-m', 'wrangle', 'prepare', '--json'], cwd=temporary, env=environment, text=True, capture_output=True)
        assert invalid.returncode == 2 and json.loads(invalid.stderr)['code'] == 'INVALID_INVOCATION'
        print(f"Verified installed wheel: {metadata['operations']} operations, {metadata['workflows']} executable workflows, console/module CLI, and sdist agent files.")


if __name__ == '__main__':
    try:
        run()
    except subprocess.CalledProcessError as error:
        sys.stderr.write(error.stderr or '')
        raise
