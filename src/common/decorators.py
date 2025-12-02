#!/usr/bin/env python

"""
Contains common use decorators
"""

import functools
import logging
import time

_TIME_DECORATORS_ENABLED = False

def set_time_decorators_enabled(enabled: bool) -> None:
    """
    Enable/disable time_fcn decorators globally, called at startup
    """
    global _TIME_DECORATORS_ENABLED
    _TIME_DECORATORS_ENABLED = enabled

def time_fcn(fn):
    """
    Decorator that measures wall-clock time of a function and logs it.
    When _TIME_DECORATORS_ENABLED is False, it becomes a simple passthrough.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Fast exit when disabled (minimal overhead)
        if not _TIME_DECORATORS_ENABLED:
            return fn(*args, **kwargs)

        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logging.info("Function %s took %.4f s", fn.__name__, elapsed)
    return wrapper
