import os

import pandas as pd; print('pandas : version {}'.format(pd.__version__))
import numpy as np; print('numpy : version {}'.format(np.__version__))
import pyarrow; print('pyarrow : version {}'.format(pyarrow.__version__))

def load_data(file_name, nrows=None):
    if os.path.exists('./data/' + file_name + '.ftr'):
        data = pd.read_feather('./data/' + file_name + '.ftr')
    else:
        data = pd.read_csv('./data/' + file_name + '.csv', sep=';', nrows=nrows)
        print('loading csv completed. Shape: ' + str(data.shape))
        data.to_feather('./data/' + file_name +'.ftr')
    return data