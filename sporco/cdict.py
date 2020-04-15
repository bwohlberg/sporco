# -*- coding: utf-8 -*-
# Copyright (C) 2015-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Constrained dictionary class."""

from builtins import str

import pprint


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



class UnknownKeyError(KeyError):
    """Exception for unrecognised dict key."""

    def __init__(self, arg):
        super(UnknownKeyError, self).__init__(arg)

    def __repr__(self):
        if isinstance(self.args[0], list):
            s = ".".join(self.args[0])
        else:
            s = str(self.args[0])
        return 'Unknown dictionary key: ' + s

    def __str__(self):
        return repr(self)



class InvalidValueError(ValueError):
    """Exception for invalid dict value."""

    def __init__(self, arg):
        super(InvalidValueError, self).__init__(arg)

    def __repr__(self):
        if isinstance(self.args[0], list):
            s = ".".join(self.args[0])
        else:
            s = str(self.args[0])
        return 'Invalid dictionary value for key: ' + s

    def __str__(self):
        return repr(self)



class ConstrainedDict(dict):
    """A dict subclass that constrains the allowed dict keys.

    Base class for a dict subclass that constrains the allowed dict
    keys, including those of nested dicts, and also initialises the
    dict with default content on instantiation. The default content is
    specified by the `defaults` class attribute, and the allowed keys
    are determined from the same attribute.
    """


    defaults = {}
    """Default content and allowed dict keys"""


    def __init__(self, d=None, pth=(), dflt=None):
        """Initialise a ConstrainedDict object.

        The object is first created with default content, which is then
        overwritten with the content of parameter `d`.  When a subdict
        is initialised via this constructor, the key path from the root
        to this subdict (i.e. the set of keys, in sequence, that select
        the subdict starting from the top-level dict) should be passed
        as a tuple via the `pth` parameter, and the defaults dict
        should be passed via the `dflt` parameter.

        Parameters
        ----------
        d : dict
          Content to overwrite the defaults
        pth : tuple of str
          Key path for objects that are subdicts of other objects
        dflt: dict
          Reference to top level defaults dict for objects that are
          subdicts of other objects
        """

        # Default arguments
        if d is None:
            d = {}
        # Initialise with empty dictionary and set path attribute (if
        # path length is zero then current object is a tree root).
        super(ConstrainedDict, self).__init__()
        self.pth = pth
        # If dflt parameter has None value then this is the top-level
        # dict in the tree and the dflt attribute should be set to the
        # class defaults attribute. Otherwise, the dflt attribute is
        # initialised with the dflt parameter.
        if dflt is None:
            self.dflt = self.__class__.defaults
        else:
            self.dflt = dflt
        # Initialise object with defaults with the corresponding node
        # (as determined by pth) in the defaults tree
        self.update(self.__class__.getnode(self.dflt, self.pth))
        # Overwrite defaults with content of parameter d
        self.update(d)



    def update(self, d):
        """Update the dict with the dict tree in parameter `d`.

        Parameters
        ----------
        d : dict
          New dict content
        """

        # Call __setitem__ for all keys in d
        for key in list(d.keys()):
            self.__setitem__(key, d[key])



    def __setitem__(self, key, value):
        """Set value corresponding to key.

        If key is a tuple, interpret it as a sequence of keys in a tree
        of nested dicts.

        Parameters
        ----------
        key : str or tuple of str
          Dict key
        value : any
          Dict value corresponding to key
        """

        # If key is a tuple, interpret it as a sequence of keys in a
        # tree of nested dicts and retrieve parent node in tree
        kc = key
        sd = self
        if isinstance(key, tuple):
            kc = key[-1]
            sd = self.__class__.getparent(self, key)
        # If value is not a dict, or if it is dict but also a
        # ConstrainedDict (meaning that it has already been
        # constructed, possibly as a derived class), or if it is a
        # dict but there is no current entry in self for the
        # corresponding key, then the value is inserted via parent
        # class __setitem__. Otherwise the value is itself a dict that
        # must be processed recursively via the update method.
        if not isinstance(value, dict) or \
           isinstance(value, ConstrainedDict) or kc not in sd:
            vc = value
            # If value is a dict but not a ConstrainedDict (if it is a
            # ConstrainedDict instance, it has already been
            # constructed, possibly as a derived class), call
            # constructor to instantiate a ConstrainedDict object
            # which becomes the value actually associated with the key
            if isinstance(value, dict) and \
               not isinstance(value, ConstrainedDict):
                # ConstrainedDict constructor is called instead of the
                # constructor of the derived class because it is
                # undesirable to force the derived class constructor to
                # have the same interface. This implies that only the root
                # node will have derived class type, and all others will
                # be of type ConstrainedDict. Since it is required that
                # all nodes use the derived class defaults class
                # attribute, it is necessary to maintain an object dflts
                # attribute that is initialised from the defaults class
                # attribute and passed down the node tree on construction.
                vc = ConstrainedDict(vc, sd.pth + (kc,), self.dflt)
            # Check that the current key and value are valid with respect
            # to the defaults tree. Relevant exceptions are caught and
            # re-raised so that the stack trace originates from this
            # method.
            try:
                sd.check(kc, vc)
            except (UnknownKeyError, InvalidValueError) as e:
                raise e
            # Call base class __setitem__ to insert key, value pair
            super(ConstrainedDict, sd).__setitem__(kc, vc)
        else:
            # Call update to handle subtree update
            sd[kc].update(value)



    def __getitem__(self, key):
        """Get value corresponding to key.

        If key is a tuple, interpret it as a sequence of keys in a tree
        of nested dicts.

        Parameters
        ----------
        key : str or tuple of str
          Dict key
        """

        # If key is a tuple, interpret it as a sequence of keys in a
        # tree of nested dicts and retrieve parent node in tree
        kc = key
        sd = self
        if isinstance(key, tuple):
            kc = key[-1]
            sd = self.__class__.getparent(self, key)
        # Return value referenced by key, or by final key in key path
        # if key is a tuple
        if kc not in sd:
            raise UnknownKeyError(key)
        return super(ConstrainedDict, sd).__getitem__(kc)



    def __str__(self):
        """Return string representation of object."""

        return pprint.pformat(self)



    def check(self, key, value):
        """Check whether `key`, `value` pair is allowed.

        The key is allowed if there is a corresponding key in the
        defaults class attribute dict.  The value is not allowed if it
        is a dict in the defaults dict and not a dict in value.

        Parameters
        ----------
        key : str or tuple of str
          Dict key
        value : any
          Dict value corresponding to key
        """

        # This test necessary to avoid unpickling errors in Python 3
        if hasattr(self, 'dflt'):
            # Get corresponding node to self, as determined by pth
            # attribute, of the defaults dict tree
            a = self.__class__.getnode(self.dflt, self.pth)
            # Raise UnknownKeyError exception if key not in corresponding
            # node of defaults tree
            if key not in a:
                raise UnknownKeyError(self.pth + (key,))
                # Raise InvalidValueError if the key value in the defaults
                # tree is a dict and the value parameter is not a dict
            elif isinstance(a[key], dict) and not isinstance(value, dict):
                raise InvalidValueError(self.pth + (key,))



    @staticmethod
    def getparent(d, pth):
        """Get the parent node of a subdict as specified by the key path
        in `pth`.

        Parameters
        ----------
        d : dict
          Dict tree in which access is required
        pth : str or tuple of str
          Dict key
        """

        c = d
        for key in pth[:-1]:
            if not isinstance(c, dict):
                raise InvalidValueError(c)
            elif key not in c:
                raise UnknownKeyError(pth)
            else:
                c = c.__getitem__(key)
        return c



    @staticmethod
    def getnode(d, pth):
        """Get the node of a subdict specified by the key path in `pth`.

        Parameters
        ----------
        d : dict
          Dict tree in which access is required
        pth : str or tuple of str
          Dict key
        """

        c = d
        for key in pth:
            if not isinstance(c, dict):
                raise InvalidValueError(c)
            elif key not in c:
                raise UnknownKeyError(pth)
            else:
                c = c.__getitem__(key)
        return c



def keycmp(a, b, pth=()):
    """Compare keys in nested dicts.

    Recurse down the tree of nested dicts `b`, at each level checking
    that it does not have any keys that are not also at the same
    level in `a`. The key path is recorded in `pth`. If an unknown key
    is encountered in `b`, an `UnknownKeyError` exception is
    raised. If a non-dict value is encountered in `b` for which the
    corresponding value in `a` is a dict, an `InvalidValueError`
    exception is raised.

    Parameters
    ----------
    a : dict
      Reference dict tree
    a : dict
      Compared dict tree
    pth : str or tuple of str
      Dict key
    """

    akey = list(a.keys())
    # Iterate over all keys in b
    for key in list(b.keys()):
        # If a key is encountered that is not in a, raise an
        # UnknownKeyError exception.
        if key not in akey:
            raise UnknownKeyError(pth + (key,))
        else:
            # If corresponding values in a and b for the same key
            # are both dicts, recursively call this method for
            # those values. If the value in a is a dict and the
            # value in b is not, raise an InvalidValueError
            # exception.
            if isinstance(a[key], dict):
                if isinstance(b[key], dict):
                    keycmp(a[key], b[key], pth + (key,))
                else:
                    raise InvalidValueError(pth + (key,))
