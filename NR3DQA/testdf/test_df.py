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
                        default=6)
    parser.add_argument('--merge', action='store_true')
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
                              "scaler":pl.Utf8}
    df = pl.DataFrame(schema=schema) 

    data_df = pl.read_csv(config.input_dir) 
    if config.merge: 
        data_df = data_df.join(pl.read_csv(config.more_feat), on='name', how='inner') 

    for scalername in valid_scalers: 
        # begin 9-folder cross data validation split
        for i in range(config.num_splits):
            scaler = getattr(sklearn.preprocessing, scalername)()
            cnt =cnt+1

            train_df = pl.read_csv(os.path.join(config.splits_dir, f'train_{i+1}.csv')) 
            test_df = pl.read_csv(os.path.join(config.splits_dir, f'test_{i+1}.csv')) 
            
            test_score = test_df.select(pl.col('mos')).to_numpy()
            test_score = test_score.reshape(len(test_score),) 
            test_set = test_df.join(data_df, on='name', how='inner').drop(['name', 'mos']).to_numpy()

            train_score = train_df.select(pl.col('mos')).to_numpy()
            train_score = train_score.reshape(len(train_score),) 
            train_set = train_df.join(data_df, on='name', how='inner').drop(['name', 'mos']).to_numpy()

            train_set = scaler.fit_transform(train_set)
            test_set = scaler.transform(test_set) 

            for i ,func in enumerate(functions): 

                func.fit(train_set, train_score) 
                predict_score = func.predict(test_set)

                # record the result
                srocc = abs(stats.spearmanr(predict_score, test_score)[0])

                df = pl.concat([df, 
                                pl.DataFrame({
                                            'function':names[i],
                                            'srocc': srocc,
                                            'scaler': scalername
                                }, 
                               schema=schema)])

    thresh = 1
    groups = df.groupby(['function', 'scaler']).agg(pl.mean(['srocc'])).sort(pl.exclude(['function', 'scaler']))
    with pl.Config(tbl_rows=-1):
        print(groups)

