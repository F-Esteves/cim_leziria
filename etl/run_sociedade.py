import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.utils import correr_pipeline, parse_args

if __name__ == "__main__":
    desde, ate = parse_args()
    correr_pipeline("Cluster 6 — Sociedade", "sociedade", desde, ate)
