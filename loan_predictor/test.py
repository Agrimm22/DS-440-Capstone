import numpy as np
import pandas as pd

file_path_part1 = 'loan_amount_dataset_part1.csv'
file_path_part2 = 'loan_amount_dataset_part2.csv'

data_part1 = pd.read_csv(file_path_part1)
data_part2 = pd.read_csv(file_path_part2)

new_data = pd.concat([data_part1, data_part2], ignore_index=True)
loan_data = pd.read_csv('loan_data (1).csv')





