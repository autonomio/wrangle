"""Run the same research workflow shipped in the installed package."""
from pathlib import Path
import runpy
import wrangle

if __name__ == "__main__":
    runpy.run_path(str(Path(wrangle.__file__).parent / "docs" / "research_batch.py"), run_name="__main__")
