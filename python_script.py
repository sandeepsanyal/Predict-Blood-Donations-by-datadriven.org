# importing modules
import pandas as pd
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

# model development
model = sm.Logit(dataset[dep_var], sm.add_constant(dataset[indep_var])) # initializing model inputs
model = model.fit() # model development
model.summary() # print model summary

# scoring dataset
result = pd.DataFrame({
    'Made Donation in March 2007':model.predict(sm.add_constant(score[indep_var])).values.tolist() # predicted probabilities
})
result.index = score['Donor ID'].values.tolist()

# exporting submission
result.to_csv(path_or_buf=directory+r"\result.csv",
              index=True)
