import os
import glob 
import argparse 
import numpy as np
import polars as pl 

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, 
                        default='./rawnp'
                        )
    config = parser.parse_args()

    main(config) 
