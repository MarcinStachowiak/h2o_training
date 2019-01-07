import h2o
from h2o.automl import H2OAutoML
h2o.init()


# Use local data file or download from GitHub
import os
docker_data_path = "data/product_backorders.csv"

# Load data into H2O
df = h2o.import_file(docker_data_path)

df.describe()

y = "went_on_backorder"
x = df.columns
x.remove(y)
x.remove("sku")

y = "went_on_backorder"
x = df.columns
x.remove(y)
x.remove("sku")

#Run AutoML
aml = H2OAutoML(max_models = 10, seed = 1)
aml.train(x = x, y = y, training_frame = df)

lb = aml.leaderboard
lb.head()
lb.head(rows=lb.nrows)


h2o.save_model(aml.leader, path = "./product_backorders_model_bin")
aml.leader.download_mojo(path = "./")