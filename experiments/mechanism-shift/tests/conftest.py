"""Make the testbed's modules importable as top-level modules (`import protocol`,
`import agents`, `import world`, ...) regardless of where pytest is launched from."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
