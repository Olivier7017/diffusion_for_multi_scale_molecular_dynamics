import pytest

try:
    from flare.bffs.sgp.calculator import SGP_Calculator  # noqa
except ImportError:
    pytest.skip("Skipping FLARE tests:  optional FLARE dependencies not installed.", allow_module_level=True)
