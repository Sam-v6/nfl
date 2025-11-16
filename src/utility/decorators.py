#!/usr/bin/env python

"""
Module: data.py
Description: Class for loading raw tracking data

Author: Syam Evani
Created: 2025-10-15
"""

def time(func):
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.4f}s")
        return result
    return wrapper