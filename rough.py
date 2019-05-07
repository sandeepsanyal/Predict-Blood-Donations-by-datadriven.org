# importing modules
import math
import pandas as pd
import random
import statsmodels.api as sm

# folder path
directory = r"C:\Users\v-sanysa\OneDrive\Github\Predict-Blood-Donations-by-datadriven.org"

# importing datasets
dataset = pd.read_csv(filepath_or_buffer=directory+r"\raw_datasets\train.csv",
                      sep=',',
                      encoding='latin-1')
score = pd.read_csv(filepath_or_buffer=directory+r"\raw_datasets\test.csv",
                    sep=',',
                    encoding='latin-1')

# preparing datasets
dataset.rename({'Unnamed: 0' : 'Donor ID'}, # fixing column name anomaly
               axis=1,
               inplace=True)
score.rename({'Unnamed: 0' : 'Donor ID'}, # fixing column name anomaly
             axis=1,
             inplace=True)

# defining variables
dep_var = 'Made Donation in March 2007'
indep_var = ['Months since Last Donation', 'Number of Donations', 'Months since First Donation']

# splitting to train & test sets
random.seed(1)
unique_id = dataset.index.T.values.tolist()
indices = random.sample(population = unique_id,
                        k = math.floor(dataset.shape[0]*0.7))
train = dataset.loc[pd.Series(unique_id).isin(indices),:]
test = dataset.loc[~pd.Series(unique_id).isin(indices),:]
del indices, unique_id

# model development
model = sm.Logit(train[dep_var], sm.add_constant(train[indep_var])) # initializing model inputs
model = model.fit() # model development
model.summary() # print model summary

# scoring on test dataset
prob_Y1_test = model.predict(sm.add_constant(test[indep_var])).values.tolist() # predicted probabilities

# scoring on score dataset
prob_Y1_score = model.predict(sm.add_constant(score[indep_var])).values.tolist() # predicted probabilities
