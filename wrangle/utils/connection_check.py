"""Compatibility name for an explicit bounded connectivity probe."""
from .._core import operation
from .network_check import network_check


@operation(returns=('scalar',), recipe='never')
def is_connected(host="www.google.com", port=80, timeout=2.0):
    """Call an explicit network probe; importing Wrangle never opens a connection."""
    return network_check(host=host, port=port, timeout=timeout)
