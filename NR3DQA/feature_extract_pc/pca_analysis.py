import os 
import numpy as np 
import pandas as pd 
import argparse 
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def df_pca_analysis(
    df: pd.DataFrame 
): 
    # You must normalize the data before applying the fit method
    scaled_features = StandardScaler().fit_transform(df.values) 
    df_normalized = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    pca = PCA(n_components=df.shape[1])
    pca.fit(df_normalized)
    print(df.shape) 

    # Reformat and view results
    loadings = pd.DataFrame(pca.components_.T,
    columns=['PC%s' % _ for _ in range(len(df_normalized.columns))],
    index=df.columns)
    print(loadings)

    plot.plot(pca.explained_variance_ratio_)
    plot.ylabel('Explained Variance')
    plot.xlabel('Components')
    plot.show()


def main(config: dict) -> None: 
    feature_data = pd.read_csv(config.input_dir,index_col = 0, keep_default_na=False)
    more_features = pd.read_csv(config.more_feat,index_col = 0, keep_default_na=False)
    merged_features = feature_data.join(more_features, on='name', how='inner')
    df_pca_analysis(feature_data) 
    df_pca_analysis(more_features) 
    df_pca_analysis(merged_features) 


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, 
                        default='../experiment for SJTU-PCQA/features.csv'
                        )
    parser.add_argument('--more-feat', type=str, 
                        default='./features/SJTU.csv'
                        )
    config = parser.parse_args()

    main(config) 
