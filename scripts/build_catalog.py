"""Generate agent navigation from the same definitions used by the recipe engine."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wrangle._catalog import catalog, operation_document

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="fail if the shipped agent documents differ from installed definitions")
args = parser.parse_args()
path = ROOT / "wrangle" / "docs" / "operations.json"
document = catalog()
content = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
readme_path = ROOT / "wrangle" / "docs" / "README.md"
readme = (ROOT / "README.md").read_text(encoding="utf-8")
readme = (readme.replace("](AGENTS.md)", "](../AGENTS.md)")
          .replace("](wrangle/docs/", "](")
          .replace('href="wrangle/docs/', 'href="')
          .replace("[MIT License](LICENSE).", "MIT License; see the distribution's license metadata."))
readme += "\n## Agent navigation\n\nThis installed manual is relative to the package root. In the operation catalog,\n`returns` identifies the output kind and `recipe.eligibility` distinguishes table\nsteps from direct calls. Read each operation's `validation` before executing it.\nAn aggregate changes the individual-record key: declare step.key for the checked\noutput observation unit.\n"
POLICIES = ('CONTRIBUTING.md', 'SECURITY.md', 'GOVERNANCE.md', 'CODE_OF_CONDUCT.md', 'ROADMAP.md', 'CHANGELOG.md')
for name in POLICIES:
    readme = readme.replace('](' + name + ')', '](project/' + name + ')')
artifacts = {path: content, readme_path: readme}
for name in POLICIES:
    artifacts[ROOT / 'wrangle' / 'docs' / 'project' / name] = (ROOT / name).read_text(encoding='utf-8').replace('](wrangle/docs/', '](../')
for entry in document["operations"]:
    artifacts[ROOT / "wrangle" / entry["manual"]] = json.dumps(operation_document(entry["name"], document), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
obsolete = set((ROOT / "wrangle" / "docs" / "operations").glob("*.json")) - set(artifacts)
if args.check:
    stale = [str(target.relative_to(ROOT)) for target, expected in artifacts.items() if not target.exists() or target.read_text(encoding="utf-8") != expected]
    stale.extend(str(target.relative_to(ROOT)) for target in sorted(obsolete))
    if stale:
        raise SystemExit("Agent documents are stale: " + ", ".join(stale) + ". Run python scripts/build_catalog.py.")
    print("Operation catalog and installed README match the execution definitions.")
else:
    for target in obsolete:
        target.unlink()
    for target, expected in artifacts.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(expected, encoding="utf-8")
    print(f"Generated catalog, installed README and {len(document['operations'])} operation documents.")
