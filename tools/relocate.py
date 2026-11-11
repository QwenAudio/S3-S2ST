"""Repository paths now resolve dynamically; retained for existing setup commands."""
from pathlib import Path
print(Path(__file__).resolve().parents[1])
