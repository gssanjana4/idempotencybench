"""Run v27_numbers.py unchanged, but write its numbers_v27.tex to TARGET and its audit
CSVs to a scratch folder, so that a check never touches the files it compares against.

  python3 analysis_v27/regen_v27.py TARGET.tex [--compare REFERENCE.tex]

With --compare, exits 1 unless TARGET is byte-identical to REFERENCE.
"""
import filecmp
import os
import runpy
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v27_common  # noqa: E402

target = os.path.abspath(sys.argv[1])
ref = sys.argv[sys.argv.index("--compare") + 1] if "--compare" in sys.argv else None
v27_common.OUT = tempfile.mkdtemp(prefix="v27out.")
_write = v27_common.Macros.write


def write(self, fn, extra=""):
    assert os.path.basename(fn) == "numbers_v27.tex", fn
    _write(self, target, extra)


v27_common.Macros.write = write
_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
try:
    runpy.run_path(os.path.join(HERE, "v27_numbers.py"), run_name="__main__")
finally:
    sys.stdout.close()
    sys.stdout = _stdout
n = sum(1 for line in open(target) if line.startswith("\\defnum{"))
if ref:
    same = filecmp.cmp(target, ref, shallow=False)
    print(f"numbers_v27.tex regenerated ({n} macros): {'byte-identical to' if same else 'DIFFERS from'} {ref}")
    sys.exit(0 if same else 1)
print(f"numbers_v27.tex regenerated ({n} macros) -> {target}")
