import polars as pl 
import numpy as np  
import argparse 
import os 

# Define the normalization function
def minmax(column):
    min_ = column['mos'].min()
    max_ = column['mos'].max() 
    df = column.with_columns(pl.lit(min_).alias('min_'))
    df = df.with_columns(pl.lit(max_).alias('max_'))
    df = df.with_columns(pl.col('mos').sub(min_).truediv(max_ - min_).alias('normalized')) 
    df = df.with_columns(pl.col('normalized').mul(5).sub(5).abs().alias('scaled'))
    df = df.with_columns(
        pl.col('normalized').lt(pl.col('normalized').shift(-1)).alias('Less_than_next')
    )
    return df


def normalize(config: dict): 
    df_metrics = pl.read_csv(config.metric) 

    # Create a series of labels for each distortion, we have 5 so [0-5)
    dis_labels = [_ for _ in range(0,5)] 
    df_labels = pl.DataFrame(
        dis_labels, 
        schema={'labels':pl.Int8}
    ) 
    # Each distortion type will be evaluated in 7 distortion levels
    # So we must repeat each label 7 times 
    df_labels = df_labels.with_columns(pl.lit(7).alias('n')) 
    df_labels = ( 
        df_labels 
        .select(
            pl.col('labels') 
            .repeat_by('n').explode()
        )
    ) 
    # We then must calculate how many times we must repeat these steps 
    repeatby = len(df_metrics)//len(df_labels)
    df_labels = pl.concat([df_labels for _ in range(repeatby)])
    # Now we add the labels to the original dataframe 
    df_metrics = pl.concat([df_metrics, df_labels], how='horizontal')
    # Apply normalization using MinMax scaling grouped by 'label'
    df_metrics = df_metrics.groupby('labels').apply(minmax)
    df_metrics = (
        df_metrics
        .with_columns(
            pl.col('name').str.split('_').arr.get(1).str.split('.').arr.get(0).cast(pl.Int64).alias('level'),
            pl.col('name').str.split('_').arr.get(0).alias('basename'),
        )
    )
    df_metrics = df_metrics.sort('basename', 'level').drop('basename', 'level') 
    with pl.Config(tbl_rows=-1):
        print(df_metrics)

    df_to_save = df_metrics.select(['name', 'scaled']).rename({'scaled': 'mos'})
    df_to_save.write_csv(os.path.join(config.output_dir, 'scaled.csv')) 

    df_to_save = df_metrics.select(['name', 'normalized']).rename({'normalized': 'mos'})
    df_to_save.write_csv(os.path.join(config.output_dir, 'normalized.csv')) 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/OurData/Distortions/metrics.csv') 
    parser.add_argument('--output-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/OurData/Distortions/')
    config = parser.parse_args()

    normalize(config) 
