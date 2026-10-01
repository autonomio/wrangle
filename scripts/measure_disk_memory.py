"""Compare peak process RAM for one reproducible local preparation workload.

Run separately from correctness tests: python scripts/measure_disk_memory.py.
macOS/Linux only; reports observed peaks, never a universal memory bound.
"""
from pathlib import Path
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile


def run(rows=100000, width=2048):
    if platform.system() not in {"Darwin", "Linux"}:
        raise SystemExit("Peak RSS measurement requires macOS or Linux resource support.")
    code = '''import json, resource, sys
from pathlib import Path
import wrangle as wr
recipe={"steps":[{"op":"cast","columns":{"value":"Int64"}},{"op":"filter","where":{"ge":[{"col":"value"},50000]},"reason":"Declared comparison cohort."}]}
result=wr.prepare(sys.argv[1],recipe,execution=sys.argv[2],output=sys.argv[3])
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != "darwin": peak *= 1024
print(json.dumps({"peak_rss_bytes":peak,"output":result.receipt["output"]}))
'''
    with tempfile.TemporaryDirectory(prefix="wrangle-memory-check-") as temporary:
        root = Path(temporary)
        source = root / "data.csv"
        suffix = "x" * width
        with source.open("w", encoding="utf-8") as stream:
            stream.write("value,text\n")
            for row in range(rows):
                stream.write(f"{row},{row:08d}{suffix}\n")
        environment = {**os.environ, "POLARS_MAX_THREADS": "2"}
        results = {}
        for mode in ("memory", "disk"):
            outcome = subprocess.run([sys.executable, "-c", code, str(source), mode, str(root / mode)], env=environment, capture_output=True, text=True, check=True)
            results[mode] = json.loads(outcome.stdout)
        assert results["memory"]["output"] == results["disk"]["output"]
        report = {"platform": platform.system(), "input_bytes": source.stat().st_size, "rows": rows, "payload_width": width, "memory_peak_rss_bytes": results["memory"]["peak_rss_bytes"], "disk_peak_rss_bytes": results["disk"]["peak_rss_bytes"], "equal_output_snapshots": True}
        print(json.dumps(report))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--width", type=int, default=2048)
    arguments = parser.parse_args()
    run(arguments.rows, arguments.width)
