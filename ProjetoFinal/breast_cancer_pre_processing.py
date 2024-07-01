import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder

breast_cancer = pd.read_csv("breast_cancer.csv")
breast_cancer.replace("?", pd.NA, inplace=True)
breast_cancer = breast_cancer.dropna()
oe = OrdinalEncoder()
breast_cancer["age"] = oe.fit_transform(breast_cancer[["age"]])
breast_cancer["menopause"] = oe.fit_transform(breast_cancer[["menopause"]])
breast_cancer["tumor-size"] = oe.fit_transform(breast_cancer[["tumor-size"]])
breast_cancer["inv-nodes"] = oe.fit_transform(breast_cancer[["inv-nodes"]])
breast_cancer["breast-quad"] = oe.fit_transform(breast_cancer[["breast-quad"]])
lb = LabelBinarizer()
breast_cancer["node-caps"] = lb.fit_transform(breast_cancer["node-caps"])
breast_cancer["breast"] = lb.fit_transform(breast_cancer["breast"])
breast_cancer["irradiat"] = lb.fit_transform(breast_cancer["irradiat"])

breast_cancer.to_csv('processed_breast_cancer.csv', index=False)
