"""Explicit bounded connectivity probe; never used during preparation or import."""
from .._core import operation
import socket
from .._core import WrangleError


@operation(returns=('scalar',), recipe='never')
def network_check(host="www.google.com", port=80, timeout=2.0):
    """Probe only when explicitly called; close the connection and honor the timeout."""
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 60:
        raise WrangleError("INVALID_ARGUMENT", "Network timeout must be in (0, 60] seconds.")
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
