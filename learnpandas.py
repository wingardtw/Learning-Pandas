import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
========================================================================================
Exploratory Analysis
- Some plots for personal use to see what to explore further
========================================================================================

"""

# import training data
df = pd.read_csv('train.csv')

credit_hist = df.Credit_History.value_counts(ascending = True)
p_table = df.pivot_table(values = 'Loan_Status', index = ['Credit_History'], aggfunc = lambda x: x.map({'Y':1, 'N':0}).mean())

print('Frequency Table for Credit History: ')
print(credit_hist)

print("\nProbability of getting a loan for each Credit History class: ")
print(p_table)

fig = plt.figure(figsize = (8,4))

ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
credit_hist.plot(ax = ax1, kind = 'bar')

ax2 = fig.add_subplot(122)
p_table.plot(ax = ax2, kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

# Tabulates counts credit_hist x loan_status
cross_tab = pd.crosstab(df.Credit_History, df.Loan_Status)
cross_tab.plot(kind = 'bar', stacked = True, color = ['red', 'blue'], grid = False)

cross_tab_2 = pd.crosstab((df.Credit_History, df.Gender), df.Loan_Status)
cross_tab_2.plot(kind = 'bar', stacked = True, color = ['red', 'blue'], grid = False)

#plt.show()

"""
========================================================================================
Data Munging
- Trying to clean our dataset of the trash
========================================================================================

"""

print("\nMissing values per column: ")
print(df.apply(lambda x: sum(x.isnull()), axis = 0))

#print("\nMissing values per row: ")
#print(df.apply(lambda x: sum(x.isnull()), axis = 1))
