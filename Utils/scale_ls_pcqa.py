import glob 
import argparse 
import polars as pl 
import os 

def main(config: dict): 
    objs = glob.glob(os.path.join(config.input_dir, "*.csv")) 
    for obj in objs: 
        df = pl.read_csv(obj) 
        df = df.with_columns(pl.col('mos').truediv(5).apply(lambda x: 1-x))
        df.write_csv(os.path.join(config.output_dir, obj))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('-i', '--input-dir', type=str, 
                        default='/home/briansenas/Documents/VQA_PC/main/database/ls_sjtu_data_info/')

    parser.add_argument('-o', '--output-dir', type=str, 
                        default='/home/briansenas/Documents/VQA_PC/main/database/ls_sjtu_data_scaled_info/') 
    config = parser.parse_args()
    main(config)
