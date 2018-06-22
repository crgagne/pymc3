import sys

import six
from . import backends_symbolic as S

__all__ = ['bool_types', 'int_types', 'float_types', 'complex_types', 'continuous_types',
           'discrete_types', 'typefilter', 'isgenerator']

bool_types = set(['int8'])

int_types = set(['int8',
                 'int16',
                 'int32',
                 'int64',
                 'uint8',
                 'uint16',
                 'uint32',
                 'uint64'])
float_types = set(['float32',
                   'float64',
                   'float32_ref',
                   'float64_ref'])
complex_types = set(['complex64',
                     'complex128'])
continuous_types = float_types | complex_types
discrete_types = bool_types | int_types

if sys.version_info[0] == 3:
    string_types = str
else:
    string_types = basestring


def typefilter(vars, types):
    # Returns variables of type `types` from `vars`
    if S.backend()=='theano':
        return [v for v in vars if v.dtype in types]
    elif S.backend()=='tensorflow':
        return [v for v in vars if v.dtype.name in types]

def isgenerator(obj):
    return ((hasattr(obj, '__next__') and six.PY3) or
            (hasattr(obj, 'next') and six.PY2))
