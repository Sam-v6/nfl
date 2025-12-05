#!/usr/bin/env python

"""
Provides timing-related decorators used across the project.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec

_TIME_DECORATORS_ENABLED = False


def set_time_decorators_enabled(enabled: bool) -> None:
	"""
	Toggles whether timing decorators emit logs.

	Inputs:
	- enabled: True to enable timing, False to silence it.

	Outputs:
	- Updates global flag controlling decorator behavior.
	"""
	global _TIME_DECORATORS_ENABLED
	_TIME_DECORATORS_ENABLED = enabled


P = ParamSpec("P")


def time_fcn[T](fn: Callable[P, T]) -> Callable[P, T]:
	"""
	Wraps a function to log its wall-clock runtime when enabled.

	Inputs:
	- fn: Function to wrap.

	Outputs:
	- wrapper: Callable that optionally measures and logs runtime.
	"""

	@functools.wraps(fn)
	def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
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
