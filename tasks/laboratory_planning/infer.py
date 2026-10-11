"""Run a frozen planning policy or JEPA world model on laboratory tasks."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lab_inference.cli import main


if __name__ == "__main__":
    main()
