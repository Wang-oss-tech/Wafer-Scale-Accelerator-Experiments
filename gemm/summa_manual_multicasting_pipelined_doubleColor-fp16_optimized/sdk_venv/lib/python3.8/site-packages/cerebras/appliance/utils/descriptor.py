# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for descriptors."""


class Descriptor:
    """
    Base class for descriptors.
    """

    _DEFAULT = object()

    def __init__(self, *, default=_DEFAULT, default_factory=_DEFAULT):
        """
        Args:
            default: The default value for the descriptor.
            default_factory: The default factory for the descriptor.
        """
        if default is not self._DEFAULT:
            self._default_factory = lambda: default
        elif default_factory is not self._DEFAULT:
            self._default_factory = default_factory
        else:
            raise ValueError(
                "Either default or default_factory must be specified"
            )

    @property
    def default(self):
        """The default value of the descriptor."""
        return self._default_factory()

    def __set_name__(self, owner, name):
        # pylint: disable=attribute-defined-outside-init
        self._name = name
        self._prefix = "_descriptor_"
        self._attr_name = self._prefix + self._name

    @property
    def name(self):
        """The name of the descriptor."""
        return self._name

    def __get__(self, obj, _type):
        if obj is None:
            return self._DEFAULT

        return getattr(obj, self._attr_name, self._DEFAULT)

    def sanitize(self, value):  # pylint: disable=no-self-use
        """
        Sanitize the value before setting it.

        Override this method to check that the provided value is valid.
        """
        return value

    def __set__(self, obj, value):
        if value is self._DEFAULT:
            value = self.default
        self.set_attr(obj, self.sanitize(value))

    def set_attr(self, obj, value):
        """Sets the actual attribute on the object.

        Override this method to customize the behavior.
        """
        setattr(obj, self._attr_name, value)
