from sklearn.model_selection import GridSearchCV

#local packages
from CML.evaluation import get_skill_scores,skill_vis,model_vis

def fit_predict_evaluate(
    options,
    pl,
    DF_train,
    DF_test,
    DF_val,
    GSparameters,
    expname,
    Savepath
    ):

    y_train = DF_train["nr_accidents"]
    X_train = DF_train.drop("nr_accidents", axis=1)

    y_val = DF_val["nr_accidents"]
    X_val = DF_val.drop("nr_accidents", axis=1)

    y_test = DF_test["nr_accidents"]
    X_test = DF_test.drop("nr_accidents", axis=1)

    if options['mode'] == 'test':
        print(f"Performing test run with default hyperparameters")

        # Fit - Predict
        print("Fitting the model...")
        pl.fit(X_train, y_train)

    elif options['mode'] == 'gridsearch':
        if GSparameters is None:
            raise(ValueError('GSparameters is not defined (models.py) for the chosen model.'))
        print("Performing a gridsearch")
        grid = GridSearchCV(pl, GSparameters, cv=5, n_jobs=-1).fit(X_train, y_train)
        best_params = grid.best_params_
        print(f' - best model parameters:{best_params}') 
        # Store the optimum model in best_pipe
        best_pipe = grid.best_estimator_
        pl = best_pipe

    # Evaluate
    print("Predicting and evaluating...")
    y_train_predict = pl.predict(X_train)
    y_val_predict = pl.predict(X_val)
    y_test_predict = pl.predict(X_test)

    D_yyhat = {
        'train':{'y':y_train, 'yhat':y_train_predict},
        'val':{'y':y_val, 'yhat':y_val_predict},
        #'test':{'y':y_test, 'yhat':y_test_predict}
        }

    scores = get_skill_scores(D_yyhat)
    
    skill_vis(D_yyhat,scores,Savepath,expname)

    model_vis(options,pl['model'],Savepath,expname)

    return scores