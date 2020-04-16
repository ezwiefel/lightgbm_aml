import argparse
import pandas as pd
import numpy as np
import pathlib
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number-of-files", "-n", type=int, default=1)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)

    args, _ = parser.parse_known_args()
    return args.__dict__

def main(number_of_files, input_path, output_path):
    print("Loading dataframe")
    df = spark.read.csv(input_path, header=True)
    
    print('Saving files')
    df.repartition(number_of_files).write.mode("overwrite").csv(output_path, header=True)


if __name__ == "__main__":
    args = parse_args()
    main(**args)