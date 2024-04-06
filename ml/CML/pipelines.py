#import packages
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

# import functions from project modules
#from CML.processors import Debugger, FeatExtractor

def get_pipelines(options,DF_N, DF_E, model_N, model_E):
    pl_preprocessor_N = build_preprocessor_pipeline(DF_N, options['features_n'])
    pl_preprocessor_E = build_preprocessor_pipeline(DF_E, options['features_e'])

    # Build the entire pipeline for nodes
    pl_N = Pipeline(
        steps = [
            ('preproc', pl_preprocessor_N),
            #('debugger', Debugger()),
            ("model", model_N)
        ]
    )

    # Build the entire pipeline for edges
    pl_E = Pipeline(
        steps = [
            ('preproc', pl_preprocessor_E),
            #('debugger', Debugger()),
            ("model", model_E)
        ]
    )

    return pl_N, pl_E
    

def build_preprocessor_pipeline(DF, featcols):
    # get the categorical and numeric column names
    num_cols = DF[featcols].select_dtypes(exclude=['object']).columns.tolist()
    cat_cols = DF[featcols].select_dtypes(include=['object']).columns.tolist()
    
    # pipeline for numerical columns
    num_pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    # pipeline for categorical columns
    cat_pipe = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', sparse=False)
    )

    # combine both the pipelines
    pl_impute_encode = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    #full preprocessor pipeline
    pl_preprocessor = Pipeline([
        ('feat_ext', FeatExtractor(cols=featcols)),
        ('impute_encode', pl_impute_encode)
    ])

    return pl_preprocessor

class Debugger(BaseEstimator, TransformerMixin):
    def transform(self, data):
        # Here you just print what you need + return the actual data. You're not transforming anything. 
        print("Shape of Pre-processed Data:", data.shape)
        return data
    def fit(self, data, y=None, **fit_params):
        # No need to fit anything, because this is not an actual  transformation. 
        return self


class FeatExtractor(object):
    def __init__(self, cols):
        self.colnames = cols
    def transform(self, X):
        Xred = X[self.colnames]
        return(Xred)
    def fit(self, X, y=None):
        return self