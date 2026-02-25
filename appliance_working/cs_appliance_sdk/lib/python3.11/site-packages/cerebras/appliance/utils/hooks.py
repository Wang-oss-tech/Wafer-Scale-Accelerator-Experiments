#
# This code is adapted from
# https://github.com/pytorch/pytorch/blob/master/torch/utils/hooks.py#L9C1-L41C40
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
#

"""Contains decorators for hooking into functions and classes."""

import weakref
from collections import OrderedDict
from functools import wraps
from inspect import isclass, isfunction
from typing import Any, Callable, Dict, Type


class RemovableHandle:
    r"""
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (dict): An additional dictionary whose keys will be deleted
            when the same keys are removed from ``hooks_dict``.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref = (
            weakref.ref(extra_dict) if extra_dict is not None else None
        )

    def remove(self):
        """Remove the hook from the corresponding hooks dictionary."""
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        if self.extra_dict_ref is not None:
            extra_dict = self.extra_dict_ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]


def hook_mixin(f: Callable):
    """
    Decorator to turn function into a mixin method for pre and post hooks.

    Two methods are added to the function: `register_pre_hook` and
    `register_post_hook`. These methods can be used to register
    hooks to the function can get executed before and after the function is called
    """

    pre_call_hooks: Dict[int, Callable] = OrderedDict()
    post_call_hooks: Dict[int, Callable] = OrderedDict()

    @wraps(f)
    def wrapper(*args, **kwargs):
        for hook in pre_call_hooks.values():
            hook(*args, **kwargs)

        result = f(*args, **kwargs)

        for hook in post_call_hooks.values():
            hook(*args, **kwargs)

        return result

    def register_pre_hook(hook: Callable) -> RemovableHandle:
        """Register a callable to be executed before the function.

        Args:
            hook: Callable to be executed before the function.
                The hook is expected to take in the same arguments as the
                original function.

        Returns:
            A handle that can be used to remove the hook by calling
            `handle.remove()`.
        """
        handle = RemovableHandle(pre_call_hooks)
        pre_call_hooks[handle.id] = hook
        return handle

    def register_post_hook(hook: Callable) -> RemovableHandle:
        """Register a callable to be executed after the function.

        Args:
            hook: Callable to be executed after the function.
                The hook is expected to take in the same arguments as the
                original function.

        Returns:
            A handle that can be used to remove the hook by calling
            `handle.remove()`.
        """
        handle = RemovableHandle(post_call_hooks)
        post_call_hooks[handle.id] = hook
        return handle

    def clear_hooks():
        """Clear all hooks."""
        # calling remove on the handle will remove it from the dict
        for handle in pre_call_hooks.values():
            handle.remove()
        for handle in post_call_hooks.values():
            handle.remove()

    wrapper.register_pre_hook = register_pre_hook
    wrapper.register_post_hook = register_post_hook
    wrapper.clear_hooks = clear_hooks
    wrapper.hook_mixin = True

    return wrapper


def pre_hook_function(*funcs):
    """Decorator to register a function to be executed before the provided functions.

    Args:
        *funcs: Functions to register the hook to. The functions must have been
            decorated with `hook_mixin` decorator.
    """
    for f in funcs:
        if not getattr(f, "hook_mixin", False):
            raise RuntimeError(
                f"Cannot hook function {f.__name__} because it is not a hook mixin"
            )

    def decorator(hook_fn):
        for f in funcs:
            f.register_pre_hook(hook_fn)
        return hook_fn

    return decorator


def post_hook_function(*funcs):
    """Decorator to register a function to be executed after the provided functions.

    Args:
        *funcs: Functions to register the hook to. The functions must have been
            decorated with `hook_mixin` decorator.
    """
    for f in funcs:
        if not getattr(f, "hook_mixin", False):
            raise RuntimeError(
                f"Cannot hook function {f.__name__} because it is not a hook mixin"
            )

    def decorator(hook_fn):
        for f in funcs:
            f.register_post_hook(hook_fn)
        return hook_fn

    return decorator


def hook_class_mixin(cls: Type):
    """Decorates all functions in a class with the hook mixins decorator"""
    if not isclass(cls):
        raise TypeError(f"Expected a class, but got: {type(cls)}")

    for attr in dir(cls):
        if attr.startswith("_") and attr not in (
            "__init__",
            "__init_subclass__",
            "__enter__",
            "__exit__",
        ):
            continue

        value = getattr(cls, attr)
        if isfunction(value):
            setattr(cls, attr, hook_mixin(value))

    return cls
