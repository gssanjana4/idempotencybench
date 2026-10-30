"""Source-checkout entry point kept for the README quickstart:

    python3 src/harness.py            # 320-task scripted grid -> results/
    python3 src/harness.py --fable5   # real-model pilot

The code lives in the idempotencybench package (src/idempotencybench/).
"""
from idempotencybench.harness import main

if __name__ == "__main__":
    main()
