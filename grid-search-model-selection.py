# Load the H2O library and start up the H2O cluter locally on your machine
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator

# Number of threads, nthreads = -1, means use all cores on your machine
# max_mem_size is the maximum memory (in GB) to allocate to H2O
h2o.init(ip="127.0.0.1",nthreads = -1)

loan_csv = "data/loan.csv"  # modify this for your machine
data = h2o.import_file(loan_csv)  # 163,987 rows x 15 columns

data.shape

data['bad_loan'] = data['bad_loan'].asfactor()  #encode the binary repsonse as a factor
data['bad_loan'].levels()  #optional: after encoding, this shows the two factor levels, '0' and '1'

# Partition data into 70%, 15%, 15% chunks
# Setting a seed will guarantee reproducibility

splits = data.split_frame(ratios=[0.7, 0.15], seed=1)

train = splits[0]
valid = splits[1]
test = splits[2]

print(train.nrow)
print(valid.nrow)
print(test.nrow)

y = 'bad_loan'
x = list(data.columns)

x.remove(y)  #remove the response
x.remove('int_rate')  #remove the interest rate column because it's correlated with the outcome

# GBM hyperparameters
gbm_params1 = {'learn_rate': [0.01, 0.1],
                'max_depth': [3, 5, 9],
                'sample_rate': [0.8, 1.0],
                'col_sample_rate': [0.2, 0.5, 1.0]}

gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          grid_id='gbm_grid1',
                          hyper_params=gbm_params1)
gbm_grid1.train(x=x, y=y,
                training_frame=train,
                validation_frame=valid,
                ntrees=100,
                seed=1)

gbm_gridperf1 = gbm_grid1.get_grid(sort_by='auc', decreasing=True)

print(gbm_gridperf1)