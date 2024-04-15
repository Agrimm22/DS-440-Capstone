import numpy as np
import pandas as pd

file_path_part1 = 'loan_amount_dataset_part1.csv'
file_path_part2 = 'loan_amount_dataset_part2.csv'

data_part1 = pd.read_csv(file_path_part1)
data_part2 = pd.read_csv(file_path_part2)

new_data = pd.concat([data_part1, data_part2], ignore_index=True)
loan_data = pd.read_csv('loan_data (1).csv')


loan_data['int.rate']




stats = loan_data.groupby('purpose')['int.rate'].agg(
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75),
    Mean='mean'
)

print(stats)