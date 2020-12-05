import os
import pytest
import numpy  as np
import tables as tb

from pandas      import DataFrame


@pytest.fixture(scope = 'session')
def FDIR():
    return os.environ['NEXTFLEX']


@pytest.fixture(scope = 'session')
def FDATA():
    return os.environ['FLEXDATA']
