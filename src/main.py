from utils import load_dataset
from preproccess import Preproccess
from model import RegressionModel

df = load_dataset()
preproccess = Preproccess(df=df)
df = preproccess.run_all()

model = RegressionModel(df=df, top_cols_num=200, degree=1)
model.train()
