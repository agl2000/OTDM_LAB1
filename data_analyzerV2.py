import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the CSV data
data = pd.read_csv('uo_nn_batch_48043775-38877082-565544_1.csv', delimiter=';', skipinitialspace=True)

#delete the last column
data = data.iloc[:, :-1]





# Create a boxplot for execution time per ISD with proper labeling
plt.figure(figsize=(12, 6))
sns.boxplot(x='isd', y='tex', data=data, palette="Set2")

# Customize the plot title and labels
plt.title('Execution Time Distribution per ISD', fontsize=16)
plt.xlabel('ISD', fontsize=14)
plt.ylabel('Execution Time (tex)', fontsize=14)

# Adjust the grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# Mapping ISD values to method names
method_mapping = {1: 'GM', 2: 'QNM', 3: 'SGM'}
data['method'] = data['isd'].map(method_mapping)

# Calculate summary statistics for each method
summary_table = data.groupby('method')['tex'].agg(['min', 'mean', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), 'max'])
summary_table.columns = ['Min', 'Mean', 'Median', 'Q1', 'Q2', 'Max']

# Display the summary table
print("Execution Time Metrics by Method")
print(summary_table)

# Plot the summary table using matplotlib
fig, ax = plt.subplots(figsize=(8, 2))  # Adjust figure size as needed
ax.axis('tight')
ax.axis('off')
ax.table(cellText=summary_table.values, colLabels=summary_table.columns, rowLabels=summary_table.index, loc='center')

plt.show()

# Sample data for mean and median of number of iterations per algorithm
iterations_summary = data.groupby('method')['niter'].agg(['mean', 'median']).reset_index()
iterations_summary.columns = ['Algorithm', 'Mean_Iterations', 'Median_Iterations']

# Now plot these values
iterations_summary = iterations_summary.melt(id_vars='Algorithm', value_vars=['Mean_Iterations', 'Median_Iterations'],
                                             var_name='Statistic', value_name='Iterations')

# Plotting
plt.figure(figsize=(8, 6))
sns.barplot(x='Algorithm', y='Iterations', hue='Statistic', data=iterations_summary, palette='viridis')
plt.title('Number of Iterations (Mean and Median) for Each Algorithm')
plt.xlabel('Algorithm')
plt.ylabel('Number of Iterations')
plt.legend(title='Statistic')
plt.show()


# Calculate mean and median for each algorithm
iterations_summary = data.groupby('method')['niter'].agg(['mean', 'median']).reset_index()
iterations_summary.columns = ['Algorithm', 'Mean Iterations', 'Median Iterations']

# Plotting the mean and median as bars
fig, ax = plt.subplots(figsize=(10, 6))
iterations_summary_melted = iterations_summary.melt(id_vars='Algorithm', value_vars=['Mean Iterations', 'Median Iterations'],
                                                    var_name='Statistic', value_name='Iterations')

# Bar plot for mean and median iterations
sns.barplot(x='Algorithm', y='Iterations', hue='Statistic', data=iterations_summary_melted, palette='viridis', ax=ax)
ax.set_title('Mean and Median Number of Iterations for Each Algorithm')
ax.set_xlabel('Algorithm')
ax.set_ylabel('Number of Iterations')

# Create the embedded table
table_data = iterations_summary.set_index('Algorithm')  # Set index for a cleaner table
table = plt.table(cellText=table_data.values,
                  colLabels=table_data.columns,
                  rowLabels=table_data.index,
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0, -0.3, 1, 0.2])  # Adjust bbox to position the table under the plot

table.auto_set_font_size(False)
table.set_fontsize(10)

# Adjust layout to make room for the table
plt.subplots_adjust(left=0.2, bottom=0.4)

# Show plot with embedded table
plt.show()

# Delete the last column if it's unwanted
data = data.iloc[:, :-1]

# Map ISD values to method names
method_mapping = {1: 'Gradient Method', 2: 'Quasi Newton Method', 3: 'Stochastic Gradient Method'}
data['method'] = data['isd'].map(method_mapping)

# Set up the figure for plotting Number of Iterations in terms of lambda
plt.figure(figsize=(15, 6))
sns.boxplot(x='la', y='niter', hue='method', data=data, palette="viridis")

# Customize plot
plt.title('Number of Iterations in Terms of Lambda for Each Method', fontsize=16)
plt.xlabel('Lambda', fontsize=14)
plt.ylabel('Number of Iterations', fontsize=14)
plt.legend(title='Method', loc='upper right')

# Show the plot
plt.show()



# Mapping ISD values to method names
method_mapping = {1: 'GM', 2: 'QNM', 3: 'SGM'}
data['method'] = data['isd'].map(method_mapping)



# Set up the plot
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Loop through each method to create a separate plot
for i, method in enumerate(['GM', 'QNM', 'SGM']):
    method_data = data[data['method'] == method]
    sns.scatterplot(
        x='niter', 
        y='tex', 
        hue='la', 
        data=method_data, 
        ax=axes[i], 
        palette="viridis",
        s=50,  # Adjust marker size for visibility
        edgecolor="w", 
        alpha=0.7
    )
    axes[i].plot([0, method_data['niter'].max()], [0, method_data['tex'].max()], color='skyblue', linestyle='-', linewidth=1)
    axes[i].set_title(f'{method} Method')
    axes[i].set_xlabel('Number of Iterations (niter)')
    if i == 0:
        axes[i].set_ylabel('Execution Time')
    else:
        axes[i].set_ylabel('')

# Display the plot
plt.tight_layout()
plt.show()



# Create an empty DataFrame to store the slope values
slopes_df = pd.DataFrame(columns=['Method', 'Slope'])
for method in ['GM', 'QNM', 'SGM']:
    method_data = data[data['method'] == method]
    X = method_data['niter']
    y = method_data['tex']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    slope = model.params['niter']
    slopes_df = pd.concat([slopes_df, pd.DataFrame({'Method': [method], 'Slope': [slope]})], ignore_index=True)
print("Table 2: Slope of the relation between Running time and iterations")
print(slopes_df)

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data (replace with your actual file path)
data = pd.read_csv('uo_nn_batch_48043775-38877082-565544.csv', delimiter=';', skipinitialspace=True)

# Map ISD values to method names
method_mapping = {1: 'GM', 2: 'QNM', 3: 'SGM'}
data['Method'] = data['isd'].map(method_mapping)

# Filter data for λ values as specified (λ = 0.01 for GM and QNM, λ = 0 for SGM)
filtered_data = data[((data['Method'] == 'GM') & (data['la'] == 0.01)) |
                     ((data['Method'] == 'QNM') & (data['la'] == 0.01)) |
                     ((data['Method'] == 'SGM') & (data['la'] == 0))]

# Calculate mean values for each metric (assuming columns 'niter', 'tex', and 'accuracy' are present)
summary_data = filtered_data.groupby('Method').agg({
    'niter': 'mean',    # Number of iterations
    'tex': 'mean',      # Execution time
    'te_acc': 'mean'  # Test accuracy
}).reset_index()

# Rename columns for clarity in the table
summary_data.columns = ['Method', 'Iterations', 'Execution time', 'Test acc.']

# Convert values to the format shown in the table
summary_data['Iterations'] = summary_data['Iterations'].round(1)
summary_data['Execution time'] = summary_data['Execution time'].round(2)
summary_data['Test acc.'] = summary_data['Test acc.'].round(2)

# Display the summary data as a table
fig, ax = plt.subplots(figsize=(6, 2))  # Adjust figure size as needed
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=summary_data.values, colLabels=summary_data.columns, rowLabels=summary_data['Method'],
                 cellLoc='center', rowLoc='center', loc='center')

# Customize font size and column width
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(summary_data.columns) + 1)))

plt.show()




##sencod part
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data (replace with your actual file path)
data = pd.read_csv('uo_nn_batch_48043775-38877082-565544.csv', delimiter=';', skipinitialspace=True)

# Map ISD values to method names
method_mapping = {1: 'GM', 3: 'QNM', 7: 'SGM'}
data['Method'] = data['isd'].map(method_mapping)

# Filter data for λ values as specified (λ = 0.01 for GM and QNM, λ = 0 for SGM)
filtered_data = data[((data['Method'] == 'GM') & (data['la'] == 0.01)) |
                     ((data['Method'] == 'QNM') & (data['la'] == 0.01)) |
                     ((data['Method'] == 'SGM') & (data['la'] == 0))]

# Calculate mean values for each metric (assuming columns 'niter', 'tex', and 'accuracy' are present)
summary_data = filtered_data.groupby('Method').agg({
    'niter': 'mean',    # Number of iterations
    'tex': 'mean',      # Execution time
    'te_acc': 'mean'  # Test accuracy
}).reset_index()

# Rename columns for clarity in the table
summary_data.columns = ['Method', 'Iterations', 'Execution time', 'Test acc.']

# Convert values to the format shown in the table
summary_data['Iterations'] = summary_data['Iterations'].round(1)
summary_data['Execution time'] = summary_data['Execution time'].round(2)
summary_data['Test acc.'] = summary_data['Test acc.'].round(2)

# Display the summary data as a table
fig, ax = plt.subplots(figsize=(6, 2))  # Adjust figure size as needed
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=summary_data.values, colLabels=summary_data.columns, rowLabels=summary_data['Method'],
                 cellLoc='center', rowLoc='center', loc='center')

# Customize font size and column width
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(summary_data.columns) + 1)))

plt.show()
