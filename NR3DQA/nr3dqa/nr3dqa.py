import polars as pl 
import numpy as np
import os 
import argparse

import sklearn.preprocessing
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso

from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats

np.random.seed(140421)

import scipy
from scipy import stats
from scipy.optimize import curve_fit

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    denominator = np.abs(bayta4) + 1e-5  # to avoid division by zero
    numerator = np.negative(X - bayta3)
    exponent = np.clip(np.divide(numerator, denominator), -500, 500)  # to avoid overflow and underflow
    logisticPart = 1 + np.exp(exponent)
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    max_val = np.max(y_label)
    min_val = np.min(y_label)
    mean_val = np.mean(y_output)
    range_val = np.max(y_output) - np.min(y_output)
    beta = [max_val, min_val, mean_val, range_val]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=1000000)
    y_output_logistic = logistic_func(y_output, *popt)
    return y_output_logistic


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, 
                        default='/home/briansenas/TFG_NRPCQA/NR3DQA/feature_extract_pc/features/OurDataPlain.csv'
                        )
    parser.add_argument('--more-feat', type=str, 
                        default='/home/briansenas/TFG_NRPCQA/NR3DQA/feature_extract_pc/features/OurData.csv'
                        )
    parser.add_argument('--splits-dir', type=str,
                        default='/home/briansenas/Desktop/PCQA-Databases/OurData/Distortions/splits')

    parser.add_argument('--num-splits', type=int,
                        default=11)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--drop-colors', action='store_true')
    config = parser.parse_args()

    cnt = 0
    valid_scalers = ["MinMaxScaler", "StandardScaler",
                     "RobustScaler", "MaxAbsScaler"]

    # initialize lists to store the evaluation metrics for each fold\n",
    random_state = np.random.randint(4294967295) 
    print(random_state)
    svm = SVR(kernel="rbf")
    neigh = KNeighborsRegressor(n_neighbors=4,weights="uniform")
    clf = DecisionTreeRegressor(random_state=random_state)
    ridge = Ridge()
    
    names = ['DecisionTree', 'SVM', 'KnnRegressor']
    functions = [clf,svm,neigh]
    schema =  {"function": pl.Utf8, 
              "srocc": pl.Float64,
               "plcc": pl.Float64, 
               "krocc": pl.Float64,
              "scaler":pl.Utf8}
    df = pl.DataFrame(schema=schema) 

    data_df = pl.read_csv(config.input_dir, infer_schema_length=1000) 
    if config.merge: 
        data_df = data_df.join(pl.read_csv(config.more_feat, infer_schema_length=1000), on='name', how='inner') 

    if config.reuse: 
        more_x = pl.read_csv('/home/briansenas/TFG_NRPCQA/NR3DQA/experiment for SJTU-PCQA/features.csv', infer_schema_length=10000)
        more_y = pl.read_csv('/home/briansenas/TFG_NRPCQA/NR3DQA/experiment for SJTU-PCQA/mos.csv', infer_schema_length=10000)
        more_y = more_y.with_row_count() 
        info = pl.DataFrame()
        for col in more_y.columns[1:]: 
            more_y = more_y.with_columns(pl.col('row_nr').apply(lambda x: f'{col}_{x}').alias(f'name'))
            info = pl.concat([info, more_y.select(f'name', col).rename({col:'mos'})], how='vertical')
        info = info.with_columns(
            pl.col('name').str.split('_').arr.get(1).cast(pl.Int64).alias('nr'),
            pl.col('name').str.split('_').arr.get(0).str.to_lowercase().alias('base')
        )
        info = info.sort('base','nr')
        info = info.with_columns(pl.col('name').str.replace('_',''))
        sjtu_data = info.join(more_x, on='name').select(pl.exclude('base','nr','row_nr'))
        sjtu_data = sjtu_data.with_columns(pl.col('mos').truediv(10))
        sjtu_data = sjtu_data.with_columns(
            pl.col(pl.Int64).cast(pl.Float64), 
            pl.col(pl.Int32).cast(pl.Float64), 
            pl.col(pl.Int16).cast(pl.Float64), 
            pl.col(pl.Int8).cast(pl.Float64), 
        )
        if config.merge: 
            test = pl.read_csv('/home/briansenas/TFG_NRPCQA/NR3DQA/feature_extract_pc/features/SJTU.csv')
            test = test.select(pl.col('^name|.*_mean|.*_std|.*_entropy.*$'))
            sjtu_data = sjtu_data.join(test, on='name')
        if config.drop_colors: 
            sjtu_data = sjtu_data.select(pl.exclude('^.l_.*|a_.*|b_.*.*$'))

    if config.drop_colors:
        data_df = data_df.select(pl.exclude('^.l_.*|a_.*|b_.*.*$'))

    for scalername in valid_scalers: 
        # begin 9-folder cross data validation split
        for sp in range(config.num_splits):
            scaler = getattr(sklearn.preprocessing, scalername)()
            cnt =cnt+1

            train_df = pl.read_csv(os.path.join(config.splits_dir, f'train_{sp+1}.csv')) 
            test_df = pl.read_csv(os.path.join(config.splits_dir, f'test_{sp+1}.csv')) 
            
            test_score = test_df.select(pl.col('mos')).to_numpy()
            test_score = test_score.reshape(len(test_score),) 
            test_set = test_df.join(data_df, on='name', how='inner').drop(['name', 'mos']).to_numpy()

            train_score = train_df.select(pl.col('mos'))
            train_set = train_df.join(data_df, on='name', how='inner').drop(['name', 'mos'])

            if config.reuse:
                train_set = pl.concat([train_set, sjtu_data.select(pl.exclude('name','mos'))])
                train_score = pl.concat([train_score, sjtu_data.select('mos')])

            train_score = train_score.to_numpy()
            train_score = train_score.reshape(len(train_score),) 
            train_set = train_set.to_numpy()

            train_set = scaler.fit_transform(train_set)
            test_set = scaler.transform(test_set) 

            for i ,func in enumerate(functions): 

                func.fit(train_set, train_score) 
                predict_score = func.predict(test_set)

                y_output_logistic = fit_function(test_score, predict_score)
                # record the result
                srocc = abs(stats.spearmanr(predict_score, test_score)[0])
                plcc = abs(stats.pearsonr(y_output_logistic, test_score)[0])
                krocc = abs(stats.kendalltau(predict_score, test_score)[0])

                df = pl.concat([df, 
                                pl.DataFrame({
                                            'function':names[i],
                                            'srocc': srocc,
                                            'plcc': plcc,
                                            'krocc': krocc,
                                            'scaler': scalername
                                }, 
                               schema=schema)])

    thresh = 1
    groups = df.groupby(['function', 'scaler']).agg(pl.mean(['srocc','plcc', 'krocc'])).sort(pl.col(['function', 'scaler']))
    with pl.Config(tbl_rows=-1):
        print(groups)

