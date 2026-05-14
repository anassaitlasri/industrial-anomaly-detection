"""Convenience wrapper: python scripts/run_feature_knn.py --data-root /path/to/mvtec"""

import sys

from mvtec_ad.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "feature-knn")
    main()
