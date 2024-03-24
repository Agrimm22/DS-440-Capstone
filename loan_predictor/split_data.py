import pandas as pd
data = pd.read_csv('loan_amount_dataset.csv')
split_index = len(data) // 2

# Split the dataset into two parts
data_first_half = data.iloc[:split_index]
data_second_half = data.iloc[split_index:]

# Save the two halves to new CSV files
data_first_half.to_csv('loan_amount_dataset_part1.csv', index=False)
data_second_half.to_csv('loan_amount_dataset_part2.csv', index=False)