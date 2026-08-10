#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from eqty_lineage.codex import build_demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = build_demo(args.output)
    print(json.dumps({"manifest": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
