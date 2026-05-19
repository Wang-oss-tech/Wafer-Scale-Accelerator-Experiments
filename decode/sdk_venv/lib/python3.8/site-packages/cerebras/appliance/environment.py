# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Module for handling environment variables available to Appliance workers."""

import os
from collections.abc import MutableMapping


class _ApplianceEnviron(MutableMapping):
    """Class for handling environment variables seen by Appliance workers.

    NOTE: Do not instantiate this class. Use the module-level `environ`
    singleton defined below instead.

    In Appliance flow, the input pipeline runs on Appliance workers, which
    are separate processes started by the Appliance on different nodes. As such,
    any environment variables set on the user process are not seen by Appliance
    workers. But since Appliance workers run user code (e.g., dataloader), it
    is desirable to be able to have some environment variables set on user side
    to be visible by workers. This class enables that.

    It provides an identical interface to `os.environ` for setting environment
    variables on user nodes. The only difference with `os.environ` is that it
    tracks any environment variables set through this class and later, Appliance
    client will package and send these variables to the Appliance workers.

    Note that this class sees variables set through `os.environ` APIs. Here's an
    example to show the behavior of this class:

        >>> appliance_environ["a"] = "1"
        >>> print(app.environ["a"])
        '1'
        >>> os.environ["a"] = "2"
        >>> print(app.environ["a"])
        '2'

    Also note that these variables are sent to the Appliance when an execute
    job is requested. Setting variables after an execute job is created has no
    effect until the next execute job is created (if any).
    """

    def __init__(self):
        """Construct an `_ApplianceEnviron` instance."""
        self._keys = set()

    def __getitem__(self, key):
        """Gets environment variable for `key`."""
        return os.environ[key]

    def __setitem__(self, key, value):
        """Sets environment variable `key` to given `value`.

        After this method, `key` is now tied to the Appliance and will be sent
        to the workers, unless it is deleted through this class's API.
        """
        os.environ[key] = value
        self._keys.add(key)

    def __delitem__(self, key):
        """Deletes environment variable `key`.

        This method also removes `key` from being tracked as an Appliance
        environment variable.
        """
        del os.environ[key]
        if key in self._keys:
            self._keys.remove(key)

    def __iter__(self):
        """Returns an iterator over Appliance environment variables."""
        return iter(self._keys)

    def __len__(self):
        """Returns number of Appliance environment variables set."""
        return len(self._keys)

    def __repr__(self):
        return 'ApplianceEnviron({{{}}})'.format(
            ', '.join(
                '{!r}: {!r}'.format(key, os.environ[key]) for key in self._keys
            )
        )

    def copy(self):
        """Returns a copy of Appliance environment variables."""
        return {key: os.environ[key] for key in self._keys}

    # pylint: disable=signature-differs
    def setdefault(self, key, default):
        """Sets environment variable `key` to `default` if not already set."""
        self._keys.add(key)
        return os.environ.setdefault(key, default)


# A singleton for handling Appliance environment variables.
appliance_environ = _ApplianceEnviron()
