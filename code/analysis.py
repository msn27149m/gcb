from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Binarizer, LabelBinarizer, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, FeatureUnion

###set parameter for plot
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (20, 13)
np.set_printoptions(precision=3)


# For data processing pipeline
class TypeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class StringIndexer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace({-1: len(s.cat.categories)}))


class CustomerizdBinarizer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return LabelBinarizer().fit(X).transform(X)


def str2datetime(x):
    try:
        return datetime.strptime(x, '%Y-%m-%d')
    except:
        return pd.NaT


def date_lag(var_list, bench):
    # asset var_list is list
    for item_date in var_list:
        name = item_date + '_' + bench
        df[name] = ((df[item_date].apply(str2datetime) - df[bench].apply(str2datetime)) / timedelta(days=31)).apply(
            np.round)
        pd.pivot_table(df, index=['churn'], values=[name], aggfunc=[np.mean, max, min, np.std]).stack(
            level=0).reset_index(level=1).reset_index().boxplot(by='churn', title=name)
        plt.savefig('../figures/' + name)


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])


if __name__ == '__main__':

    # Load Data
    df_train = pd.read_csv('/Users/CC/Desktop/imbalance/BCG/data/ml_case_training_data.csv', encoding='utf-8')
    df_hist = pd.read_csv('/Users/CC/Desktop/imbalance/BCG/data/ml_case_training_hist_data.csv', encoding='utf-8')
    df_output = pd.read_csv('/Users/CC/Desktop/imbalance/BCG/data/ml_case_training_output.csv', encoding='utf-8')

    # Meger data with output
    df = pd.merge(df_train, df_output, left_on='id', right_on='id', how='inner')

    # stack the hist data into data
    df_hist_pivot = pd.pivot_table(df_hist, index=['id'], columns=['price_date'], fill_value=0)

    # meger hist data into dataframe
    for col, date in zip(df_hist_pivot.columns.get_level_values(0), df_hist_pivot.columns.get_level_values(1)):
        new_name = str(col) + "_" + str(date)
        tmp = df_hist_pivot[col][date].to_frame(name=new_name).reset_index()
        df = pd.merge(df, tmp, left_on='id', right_on='id', how='outer')

    # Categorize data for future processing
    cat_feat_list = ['activity_new', 'campaign_disc_ele', 'channel_sales', 'has_gas', 'nb_prod_act', 'num_years_antig',
                     'origin_up']

    num_feat_list = ['cons_12m', 'cons_gas_12m', 'cons_last_month', 'forecast_base_bill_ele', 'forecast_base_bill_year',
                     'forecast_bill_12m', 'forecast_cons', \
                     'forecast_cons_12m', 'forecast_cons_year', 'forecast_discount_energy', 'forecast_meter_rent_12m',
                     'forecast_price_energy_p1', 'forecast_price_energy_p2',
                     'forecast_price_pow_p1', 'imp_cons', 'margin_gross_pow_ele', 'margin_net_pow_ele', 'net_margin',
                     'pow_max']

    opt_feat_list = ['id', 'price_p1_fix_2015-01-01', 'price_p1_fix_2015-02-01', 'price_p1_fix_2015-03-01',
                     'price_p1_fix_2015-04-01', 'price_p1_fix_2015-05-01', 'price_p1_fix_2015-06-01', \
                     'price_p1_fix_2015-07-01', 'price_p1_fix_2015-08-01', 'price_p1_fix_2015-09-01',
                     'price_p1_fix_2015-10-01', 'price_p1_fix_2015-11-01', 'price_p1_fix_2015-12-01', \
                     'price_p1_var_2015-01-01', 'price_p1_var_2015-02-01', 'price_p1_var_2015-03-01',
                     'price_p1_var_2015-04-01', 'price_p1_var_2015-05-01', 'price_p1_var_2015-06-01', \
                     'price_p1_var_2015-07-01', 'price_p1_var_2015-08-01', 'price_p1_var_2015-09-01',
                     'price_p1_var_2015-10-01', 'price_p1_var_2015-11-01', 'price_p1_var_2015-12-01', \
                     'price_p2_fix_2015-01-01', 'price_p2_fix_2015-02-01', 'price_p2_fix_2015-03-01',
                     'price_p2_fix_2015-04-01', 'price_p2_fix_2015-05-01', 'price_p2_fix_2015-06-01', \
                     'price_p2_fix_2015-07-01', 'price_p2_fix_2015-08-01', 'price_p2_fix_2015-09-01',
                     'price_p2_fix_2015-10-01', 'price_p2_fix_2015-11-01', 'price_p2_fix_2015-12-01', \
                     'price_p2_var_2015-01-01', 'price_p2_var_2015-02-01', 'price_p2_var_2015-03-01',
                     'price_p2_var_2015-04-01', 'price_p2_var_2015-05-01', 'price_p2_var_2015-06-01', \
                     'price_p2_var_2015-07-01', 'price_p2_var_2015-08-01', 'price_p2_var_2015-09-01',
                     'price_p2_var_2015-10-01', 'price_p2_var_2015-11-01', 'price_p2_var_2015-12-01', \
                     'price_p3_fix_2015-01-01', 'price_p3_fix_2015-02-01', 'price_p3_fix_2015-03-01',
                     'price_p3_fix_2015-04-01', 'price_p3_fix_2015-05-01', 'price_p3_fix_2015-06-01', \
                     'price_p3_fix_2015-07-01', 'price_p3_fix_2015-08-01', 'price_p3_fix_2015-09-01',
                     'price_p3_fix_2015-10-01', 'price_p3_fix_2015-11-01', 'price_p3_fix_2015-12-01', \
                     'price_p3_var_2015-01-01', 'price_p3_var_2015-02-01', 'price_p3_var_2015-03-01',
                     'price_p3_var_2015-04-01', 'price_p3_var_2015-05-01', 'price_p3_var_2015-06-01', \
                     'price_p3_var_2015-07-01', 'price_p3_var_2015-08-01', 'price_p3_var_2015-09-01',
                     'price_p3_var_2015-10-01', 'price_p3_var_2015-11-01', 'price_p3_var_2015-12-01']

    date_feat_list = ['date_activ', 'date_end', 'date_first_activ', 'date_modif_prod', 'date_renewal']

    # Data quality check:
    cat_feat_drop = df[cat_feat_list].columns[
        (df[cat_feat_list].isnull().sum() / df[cat_feat_list].isnull().count() > 0.3)].tolist()
    num_feat_drop = df[num_feat_list].columns[
        (df[num_feat_list].isnull().sum() / df[num_feat_list].isnull().count() > 0.3)].tolist()
    feat_drop = cat_feat_drop + num_feat_drop + opt_feat_list
    df.drop(columns=feat_drop, axis=1, inplace=True)

    # filling missing value
    # Numerical data fill missing value
    from sklearn.preprocessing import Imputer

    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    num_feat_exist = list(set(num_feat_list) - set(num_feat_drop))
    df[num_feat_exist] = pd.DataFrame(imp.fit_transform(df[num_feat_exist]), columns=num_feat_exist)

    # Set categrical data type for future processing
    cat_feat_exist = list(set(cat_feat_list) - set(cat_feat_drop))
    df[cat_feat_exist] = df[cat_feat_exist].astype('category')

    # Anomaly detection and remove
    # TODO

    # Pipeline for features and target
    num_pipe = Pipeline([('selector', TypeSelector(np.number)), ('scaler', StandardScaler())])
    cate_pipe = Pipeline([('selector', TypeSelector('category')), ('category', StringIndexer()),
                          ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    full_pipeline = FeatureUnion(transformer_list=[('cate_pipeline', cate_pipe), ('num_pipeline', num_pipe)])
    df_y = df['churn'].copy()
    y = df_y.values
    df.drop('churn', axis=1, inplace=True)
    x = full_pipeline.fit_transform(df[cat_feat_exist + num_feat_exist])
    print('Data pipeline has done')
    #   comparison between random sampling and under-sampling
    '''
    #   random sampling 
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))
    df_full = pd.concat([df[num_feat_exist], df_y], axis=1)
    corr = df_full.corr()
    corr['churn'].plot(kind='bar', ax=ax1)

    #   under-sampling
    df_under = df_full.sample(frac=1)
    df_under_churn = df_under.loc[df_under['churn'] == 1]
    df_under_not_churn = df_under.loc[df_under['churn'] == 0][:1595]
    df_under_full = pd.concat([df_under_churn, df_under_not_churn])
    df_under_sampling= df_under_full.sample(frac=1, random_state=42)
    under_corr = df_under_sampling.corr()
    #sns.heatmap(under_corr, cmap='coolwarm_r', annot_kws={'size':20})
    under_corr['churn'].plot('bar', ax=ax2)

    '''

    '''
    TODO delete 
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    '''

    # Linear Model
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
    from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score, \
        precision_recall_curve, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

    logis_accuracy_lst = []
    logis_precision_lst = []
    logis_recall_lst = []
    logis_f1_lst = []
    logis_auc_lst = []




    split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('Stratified split done')

        polybig_features = PolynomialFeatures(degree=1, include_bias=False)

        print('Polynomial Done')

        sm = SMOTE(ratio='minority')
        sm_train_x, sm_train_y = sm.fit_sample(X_train, y_train)

        '''
        polybig_features = PolynomialFeatures(degree=1, include_bias=False)

        log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        log_reg = LogisticRegression( class_weight='balanced', max_iter=50000, n_jobs=-1, verbose=True)
        grid_log_reg = GridSearchCV(log_reg, log_reg_params)
        #rand_log_reg = RandomizedSearchCV(log_reg, log_reg_params, n_iter=4, verbose=2)
        #pipeline = imbalanced_make_pipeline(SMOTE(ratio='minority'), grid_log_reg)
        #logis_model = pipeline.fit(X_train, y_train)
        grid_log_reg.fit(sm_train_x, sm_train_y)

        logis_best_est = grid_log_reg.best_estimator_
        print(grid_log_reg.best_params_)
        #logis_y_predict = logis_best_est.predict(X_test)
        precisions, recalls, thresholds = precision_recall_curve(y_test, logis_best_est.predict_proba(X_test)[:, 1])
        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        plt.show()
        print('logistic Done')
        '''
        from sklearn.svm import SVC

        svm_clf = SVC(max_iter=50000, verbose=True)
        svc_params = {'degree':[1,2], 'C': [0.1, 0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
        grid_svc = GridSearchCV(svm_clf, svc_params)
        grid_svc.fit(sm_train_x, sm_train_y)
        svc_best_est = grid_svc.best_estimator_
        print(grid_svc.best_params_)

    '''
        logis_accuracy_lst.append(pipeline.score(X_test, y_test))
        logis_precision_lst.append(precision_score(y_test, logis_y_predict))
        logis_recall_lst.append(recall_score(y_test, logis_y_predict))
        logis_f1_lst.append(f1_score(y_test, logis_y_predict))
        logis_auc_lst.append(roc_auc_score(y_test, logis_y_predict))
    
    smote_prediction = logis_best_est.predict(X_test)
    print(classification_report(y_test, smote_prediction, target_names=['0', '1']))
    # precisions, recalls, thresholds = precision_recall_curve(, y_scores)
    print(logis_accuracy_lst)
    print(logis_precision_lst)
    precisions, recalls, thresholds = precision_recall_curve(y_test, logis_y_predict)

    logis_reg_clf = confusion_matrix(y_test, logis_y_predict)
    #sns.heatmap(logis_reg_clf, annot=True, cmap=plt.cm.copper)

    ############################################################################################

    # SVC
    # TODO
    from sklearn.svm import SVC
    svm_clf = SVC()
    svc_params = {'C': [0.1, 0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    grid_svc = GridSearchCV(svm_clf, svc_params)

    svc_accuracy_lst = []
    svc_precision_lst = []
    svc_recall_lst = []
    svc_f1_lst = []
    svc_auc_lst = []

    for train_index, test_index in split.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pipeline = imbalanced_make_pipeline(SMOTE(ratio='minority'), grid_svc)
        svc_model = pipeline.fit(X_train, y_train)
        best_svc_est = grid_svc.best_estimator_
        y_predict = best_svc_est.predict(X_test)

    svc_clf_prediction = best_svc_est.predict(X_test)
    print(classification_report(y_test, svc_clf_prediction, target_names=['0', '1']))
    # precisions, recalls, thresholds = precision_recall_curve(, y_scores)
    print(svc_accuracy_lst)
    print(svc_precision_lst)




    # RandomForest
    # TODO
    tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)),  "min_samples_leaf": list(range(5,7,1))}
    grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
    grid_tree.fit(X_train, y_train)
    
    
    # Voting
    # TODO
    voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

'''
