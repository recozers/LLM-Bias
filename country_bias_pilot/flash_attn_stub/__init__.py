"""Stub flash_attn — satisfies DeepSeek's import check.

Actual attention uses eager implementation (attn_implementation='eager'),
so these functions are never called.
"""


def flash_attn_func(*args, **kwargs):
    raise RuntimeError("flash_attn stub — use attn_implementation='eager'")


def flash_attn_varlen_func(*args, **kwargs):
    raise RuntimeError("flash_attn stub — use attn_implementation='eager'")
