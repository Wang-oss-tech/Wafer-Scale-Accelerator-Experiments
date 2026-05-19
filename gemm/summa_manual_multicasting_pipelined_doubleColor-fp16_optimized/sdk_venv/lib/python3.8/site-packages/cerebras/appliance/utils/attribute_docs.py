# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
"""
Utilities to extract PEP224 style attribute docstrings to __doc__
"""

import ast
import inspect
from io import StringIO
from typing import Optional, Tuple


def ast_find_classdef(tree):
    """
    Return the first ClassDef type in the given AST.
    """
    for e in ast.walk(tree):
        if isinstance(e, ast.ClassDef):
            return e
    return None


def _unparse(n):
    """
    Minimal implementation of ast.unparse for Python 3.8
    """
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Subscript):
        return f"{_unparse(n.value)}[{_unparse(n.slice)}]"
    if isinstance(n, ast.Tuple):
        return ",".join([_unparse(x) for x in n.elts])
    if isinstance(n, ast.Index):
        return _unparse(n.value)
    if isinstance(n, ast.Constant):
        return n.value
    raise ValueError(f"Unhandled AST type {n}")


def attribute_docs(cls):
    """
    Enable attribute documentations for (data)classes. Use this function as a
    decorator.

    ```
    @attribute_docs
    @dataclass
    class TargetCoder:
        "Target coder for object detection"

        num_classes: int
        "Number of detection classes"

        min_size: float
        "Normalized minimum bounding box size"
    ```
    """
    # == find the class defining syntax tree ==
    src = inspect.getsource(cls)
    tree = ast.parse(src)
    tree = ast_find_classdef(tree)

    # == gather attribute doc strings ==
    # * We skip the first expr, because it is either a class docstring or something else
    # * The idea is that docstring appears on top of the attribute.
    # * Therefore, we search for any string node, mark that as a docstring.
    # * If a class attribute define node appears after the docstring, we store the docstring
    #    along with the class attribute's information
    attribute_typedocs = {}
    last_attr: Optional[Tuple[str, str]] = None
    for expr in tree.body[1:]:
        # if this next node is an annotation,
        if isinstance(expr, ast.AnnAssign):
            # expr.target is a ast.Name
            name = expr.target.id
            type_name = _unparse(expr.annotation)
            last_attr = (name, type_name)

        # When encouter an Expr, check if the expr a string
        if last_attr and isinstance(expr, ast.Expr):
            # The value is a ast.Value node
            # therefore another access to value is needed
            value = expr.value.value
            if isinstance(value, str):
                docstring = value.strip()
                attr_name, attr_type = last_attr
                last_attr = None
                attribute_typedocs[attr_name] = (attr_type, docstring)

    # == Append to the class documentation ==
    # * if there is no attribute docstring, leave it be
    if len(attribute_typedocs) > 0:
        old_docs = cls.__doc__
        append_docs = build_attibute_docstrings(attribute_typedocs)
        cls.__doc__ = f"""{old_docs}\n\n{append_docs}"""
        cls.__attr_doc__ = attribute_typedocs
    return cls


def build_attibute_docstrings(docs: dict):
    """
    Construct a Google-style Attribute docstring block from the given extracted
    docstrings from the classes attributes.
    """
    # Create pretty formatting for the attribute docs
    with StringIO() as io:
        io.write("Attributes:\n")
        for var_name, (type_name, docstring) in docs.items():
            # == Multiline vs inline doc format ==
            # * if the doc is inline, simply use the `x (type): docstring`
            # * if the doc is multiline, create a new paragraph
            if "\n" in docstring:
                lines = docstring.split("\n")
                lines = ["\t\t" + line.strip() for line in lines]
                docstring = "\n".join(lines)
                line = f"\t{var_name} ({type_name}):\n{docstring}\n"
            else:
                line = f"\t{var_name} ({type_name}): {docstring}\n"

            # Add the docstring line
            io.write(line)
        io.seek(0)
        docstring = io.read()
    return docstring
