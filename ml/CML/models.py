from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

def construct_model(opt):
    if opt == "LR":
        model = LinearRegression()
        GSparameters = {
                         'fit_intercept': [True, False]
                        }
    elif opt == "DTR":
        model_N = DecisionTreeRegressor(
            random_state=420,
            min_impurity_decrease=0.015,
            max_depth=6,
            min_samples_split=30
            )
        model_E = DecisionTreeRegressor(
            random_state=420,
            min_impurity_decrease=0.005,
            max_depth=6,
            min_samples_split=100
            )
        GSparameters_N = {
                         'model__min_impurity_decrease': [0.001, 0.0015, 0.002],
                         'model__max_depth': [5, 6, 7],
                         'model__min_samples_split': [20, 30, 40],
                        }
        GSparameters_E = {
                         'model__min_impurity_decrease': [0.00025, 0.0005, 0.0075],
                         'model__max_depth': [4, 6, 8],
                         'model__min_samples_split': [75, 100, 125],
                        }
    elif opt == "XGBR":
        model_N = XGBRegressor(
            random_state=420,
            eta=0.03,
            n_estimators=100,
            max_depth=6,
        )
        model_E = XGBRegressor(
            random_state=420,
            eta=0.03,
            n_estimators=150,
            max_depth=6,
        )
        GSparameters_N = {
                         'model__eta':[0.01,0.03,0.05],
                         'model__n_estimators': [50,150,200],
                         #'model__max_depth': [3,5,7]
                        }
        GSparameters_E = {
                         'model__eta':[0.01,0.03,0.05],
                         'model__n_estimators': [50,150,200],
                         #'model__max_depth': [3,5,7],
                        }
    else:
        raise (ValueError(f"model:{opt} option not defined"))

    return model_N,model_E,GSparameters_N,GSparameters_E
