from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

categorical = ['cod_prov','indfall','segmento', 'nomprov', 'ind_actividad_cliente' ,'canal_entrada', 'ind_empleado', 'pais_residencia', 'sexo', 'ind_nuevo', 'indrel', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext']
to_drop = ['fecha_alta','conyuemp', 'ult_fec_cli_1t', 'ncodpers', 'fecha_dato']


class FeatureExtractor(TransformerMixin):
    def __init__(self):
        self.imp = Imputer(strategy = 'mean')
        pass
    
    def fit(self, X_df, y=None):
        x_df = X_df.drop(to_drop, axis = 1)
        x_df = pd.get_dummies(x_df, columns=categorical)
        x_df['antiguedad'] = x_df['antiguedad'].apply(lambda s : np.nan if s == '     NA' else s)
        x_df['age'] = x_df['age'].apply(lambda s : np.nan if s == ' NA' else s)
        self.columns = x_df.columns
        self.imp.fit(x_df)
        return self
 
    def fit_transform(self, X_df, y=None):
        self.fit(X_df)
        return self.transform(X_df)
 
    def transform(self, X_df):
        x_df = X_df.drop(to_drop, axis = 1)
        x_df = pd.get_dummies(x_df, columns=categorical)
        x_df['antiguedad'] = x_df['antiguedad'].apply(lambda s : np.nan if s == '     NA' else s)
        x_df['age'] = x_df['age'].apply(lambda s : np.nan if s == ' NA' else s)
        x_df = x_df.loc[:, [f for f in x_df.columns if f in self.columns]]
        x_df = x_df.loc[:, self.columns]
        x_df = self.imp.transform(x_df)
        return x_df