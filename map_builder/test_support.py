from __future__ import annotations

import os
import unittest


metashape_functional_test = unittest.skipUnless(
    os.environ.get("RUN_METASHAPE_FUNCTIONAL_TESTS") == "1",
    "set RUN_METASHAPE_FUNCTIONAL_TESTS=1 to run large Metashape functional tests",
)
