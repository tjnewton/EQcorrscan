"""
:copyright:
    EQcorrscan developers.

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import sys
import importlib
import warnings

__all__ = ['template_gen', 'match_filter', 'bright_lights', 'lag_calc',
           'subspace']

_import_map = {}

_depreciated = ['template_gen.from_sfile', 'template_gen.from_contbase']


class EQcorrscanDeprecationWarning(UserWarning):
    """
    Force pop-up of warnings.
    """
    pass


class EQcorrscanRestructureAndLoad(object):
    """
    Path finder and module loader for transitioning
    """
    def find_module(self, fullname, path=None):
        # Compatibility with namespace paths.
        if hasattr(path, "_path"):
            path = path._path

        if not path or not path[0].startswith(__path__[0]):
            return None

        for key in _import_map.keys():
            if fullname.startswith(key):
                break
        else:
            return None
        return self

    def load_module(self, name):
        # Use cached modules.
        if name in sys.modules:
            return sys.modules[name]
        # Otherwise check if the name is part of the import map.
        elif name in _import_map:
            new_name = _import_map[name]
        elif name in _depreciated:
            raise ImportError(
                "%s is no longer supported, use template_gen.from_meta_file" %
                name)
        else:
            new_name = name
            for old, new in _import_map.items():
                if not new_name.startswith(old):
                    continue
                new_name = new_name.replace(old, new)
                break
            else:
                return None

        # Don't load again if already loaded.
        if new_name in sys.modules:
            module = sys.modules[new_name]
        else:
            module = importlib.import_module(new_name)

        # Warn here as at this point the module has already been imported.
        warnings.warn("Module '%s' is deprecated and will stop working "
                      "with the next EQcorrscan version. Please import module "
                      "'%s' instead." % (name, new_name),
                      EQcorrscanDeprecationWarning)
        sys.modules[new_name] = module
        sys.modules[name] = module
        return module


sys.meta_path.append(EQcorrscanRestructureAndLoad())

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
