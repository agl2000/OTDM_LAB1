import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('uo_nn_batch_48043775-38877082-565544.csv', delimiter=';', skipinitialspace=True)

#delete the last column
data = data.iloc[:, :-1]



print(data)

# # Rename columns for easier access
data.columns = ['num_target', 'lambda', 'isd', 'niter', 'tex', 'tr_acc', 'te_acc', 'L*']



# Create a boxplot for training accuracy
plt.figure(figsize=(12, 6))
sns.boxplot(x='lambda', y='tr_acc', hue='isd', data=data)
plt.title('Training Accuracy by Lambda and ISD')
plt.xlabel('Lambda')
plt.ylabel('Training Accuracy')
plt.legend(title='ISD')
plt.show()

# Create a boxplot for test accuracy
plt.figure(figsize=(12, 6))
sns.boxplot(x='lambda', y='te_acc', hue='isd', data=data)
plt.title('Test Accuracy by Lambda and ISD')
plt.xlabel('Lambda')
plt.ylabel('Test Accuracy')
plt.legend(title='ISD')
plt.show()

# Create a boxplot for executuin time
plt.figure(figsize=(12, 6))
sns.boxplot(x='lambda', y='tex', hue='isd', data=data)
plt.title('Execution Time by Lambda and ISD')
plt.xlabel('Lambda')
plt.ylabel('Execution Time')
plt.legend(title='ISD')
plt.show()

# Create a boxplot for number of iterations
plt.figure(figsize=(12, 6))
sns.boxplot(x='lambda', y='niter', hue='isd', data=data)
plt.title('Number of Iterations by Lambda and ISD')
plt.xlabel('Lambda')
plt.ylabel('Number of Iterations')
plt.legend(title='ISD')
plt.show()