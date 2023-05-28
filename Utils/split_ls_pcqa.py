import polars as pl 
import numpy as np 
import os 
import argparse 

def main(config: dict): 
    df = pl.read_excel(config.input_dir) 

    # Define the distortions I will be evaluating: 
    distortions = [
        "Octree", "LocalOffset", "LocalRotation", "DownSampling", "GaussianShifting"
    ]

    # From the whole name_distortion_level.ply get the basename 
    df = ( 
        df.with_columns( 
            pl.col('name').str.split('_').arr.get(0).alias('basename'), 
            pl.col('name').str.split('_').arr.get(1).alias('distortion') 
        )
        .filter(
            pl.col('distortion').is_in(distortions)
        ) 
    ) 

    # Generate the array of unique names 
    gdf = df.select('basename').groupby('basename').n_unique().to_numpy().flatten()  
    # 104 / 4 = 26 -> generate a 4-fold train/test to choose the best model
    splits = np.array_split(gdf, 4)

    for i, split in enumerate(splits): 
        tmp_train = df.filter(~pl.col('basename').is_in(split.tolist()))
        tmp_test = df.filter(pl.col('basename').is_in(split.tolist())) 

        tmp_train_path = os.path.join(config.output_dir, f'train_{i+1}.csv')
        tmp_test_path = os.path.join(config.output_dir, f'test_{i+1}.csv')

        tmp_train = tmp_train.drop('basename') 
        tmp_test = tmp_test.drop('basename') 

        tmp_train.write_csv(tmp_train_path) 
        tmp_test.write_csv(tmp_test_path) 
     

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('-i', '--input-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/LS-SJTU-PCQA/pseudoMOS.xlsx') 

    parser.add_argument('-o', '--output-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/LS-SJTU-PCQA/splits') 
    parser.add_argument('-d', action='store_true') 
    config = parser.parse_args()

    main(config)
