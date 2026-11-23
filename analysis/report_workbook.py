"""Render the saved strategy report tables into an Excel workbook with editable charts."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


def export_workbook(directory: Path, *, previews: Path | None = None):
    # Desktop supplies the Excel renderer. Other installations can supply the same Node package.
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    modules = Path(os.environ.get("TLH_NODE_MODULES", bundled / "node_modules")).resolve()
    node = os.environ.get("TLH_NODE") or str(bundled / "bin/node")
    if not Path(node).is_file():
        node = shutil.which("node")
    if not node or not (modules / "@oai/artifact-tool").exists():
        print("Excel export requires Node and @oai/artifact-tool. The three report CSVs are available.",
              file=sys.stderr)
        return None
    tables = {}
    for name in ("daily", "periods", "benchmarks"):
        with (directory / f"report_{name}.csv").open(newline="") as f:
            tables[name] = list(csv.DictReader(f))
    with TemporaryDirectory(prefix=".tlh-report-", dir=directory.resolve()) as tmp:
        stage = Path(tmp)
        (stage / "node_modules").symlink_to(modules, target_is_directory=True)
        script = stage / "strategy_workbook.mjs"
        shutil.copyfile(Path(__file__).with_name(script.name), script)
        data = stage / "report.json"
        data.write_text(json.dumps(tables, ensure_ascii=False), encoding="utf-8")
        workbook = directory / "strategy_report.xlsx"
        rendered = stage / workbook.name
        args = [node, str(script), str(data), str(rendered)]
        if previews is not None:
            args.append(str(previews.resolve()))
        subprocess.run(args, check=True)
        rendered.replace(workbook)
    return workbook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--previews", type=Path)
    args = parser.parse_args()
    if export_workbook(args.directory, previews=args.previews) is None:
        raise SystemExit(1)
