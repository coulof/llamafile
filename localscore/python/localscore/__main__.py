"""Entry point for running localscore as a module: python -m localscore"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
