"""python3 -m idempotencybench <command> (same as the `idempotencybench` script)."""
import sys

from .cli import main

sys.exit(main())
