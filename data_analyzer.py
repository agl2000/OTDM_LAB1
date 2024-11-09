import pandas as pd

# Load the CSV data
data = pd.read_csv('uo_nn_batch_123456-789101-565544.csv', delimiter=';', skipinitialspace=True)

#delete the last column
data=data[,1:8]
print(data)

# # Rename columns for easier access
data.columns = ['num_target', 'lambda', 'isd', 'niter', 'tex', 'tr_acc', 'te_acc', 'L*']


# # Group by different combinations of num_target, lambda, and isd to compute mean metrics
# results_summary = data.groupby(['num_target', 'lambda', 'isd']).agg({
#     'niter': 'mean',
#     'tex': 'mean',
#     'tr_acc': 'mean',
#     'te_acc': 'mean',
#     'L*': 'mean'
# }).reset_index()

# data

# results_summary