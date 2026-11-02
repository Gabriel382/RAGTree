#!/usr/bin/env python3
"""Three-protocol KBS journal runner. Execute from the repository root."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ragtree.experiments.kbs.cli import main
if __name__=='__main__':raise SystemExit(main())
