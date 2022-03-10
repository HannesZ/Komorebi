import imp

from datetime import datetime
from utils import convert_categoricals

import os
from lightGBM import *
from data import load_data

now = datetime.now()

data = load_data('A202001', nrows=10000000)

# severity-model:
info_cols = []
model_name = now.strftime('severity_model' + '-v%Y-%m-%dT%H-%M-%S.xlsx')

cat_cols = ['ORG_CLE_REG', 'BEN_RES_REG',
       'BEN_CMU_TOP', 'BEN_QLT_COD', 'BEN_SEX_COD']

"""        #, 'DDP_SPE_COD',
       'ETE_CAT_SNDS', 'ETE_REG_COD', 'ETE_TYP_SNDS', 'ETP_REG_COD',
       'ETP_CAT_SNDS', 'MDT_TYP_COD', 'MFT_COD', 'PRS_FJH_TYP', 'PRS_ACT_COG',
       'PRS_ACT_NBR', 'PRS_ACT_QTE', 'PRS_REM_BSE', 'SOI_ANN', 'SOI_MOI', 'ASU_NAT', 
       'ATT_NAT', 'CPT_ENV_TYP', 'DRG_AFF_NAT', 'ETE_IND_TAA', 'EXO_MTF', 
       'MTM_NAT', 'PRS_NAT', 'PRS_PPU_SEC',
       'PRS_PDS_QCP', 'EXE_INS_REG', 'PSE_ACT_SNDS', 'PSE_ACT_CAT', 'PSE_SPE_SNDS',
       'PSE_STJ_SNDS', 'PRE_INS_REG', 'PSP_ACT_SNDS', 'PSP_ACT_CAT',
       'PSP_SPE_SNDS', 'PSP_STJ_SNDS' """
num_cols = ['AGE_BEN_SNDS']

exclude = ['FLT_ACT_COG', 'FLT_ACT_NBR',
       'FLT_ACT_QTE', 'FLT_PAI_MNT', 'FLT_DEP_MNT', 'FLT_REM_MNT', 'CPL_COD', 'PRS_REM_TAU', 'PRS_REM_TYP', 'PRS_DEP_MNT']

unknown_cols = ['TOP_PS5_TRG', 'Unnamed: 55']

target = 'PRS_REM_MNT'

data[target] = data[target].astype('float64')

data = convert_categoricals(data_frame=data, cols=cat_cols)

sev_lgbm = LGBMRegressorModel(data=data, target=target, cat_cols=cat_cols, num_cols=num_cols, param_grid=default_LGBMRegressor_param_grid, model_name=model_name).train(tune=True, export=True, info_cols=info_cols)


