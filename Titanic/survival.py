# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#Accuracy 0.78
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train=pd.read_csv("../input/train.csv") 
test=pd.read_csv("../input/test.csv")
train["Child"] = float('NaN')



train["Child"][train["Age"]>=18]=0
train["Child"][train["Age"]<18]=1

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

train["Age"] = train["Age"].fillna(train.Age.median())
test["Age"]=test["Age"].fillna(test.Age.median())

train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Embarked"] = test["Embarked"].fillna("S")
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

test["Fare"]=test["Fare"].fillna(test.Fare.mean())

target = train["Survived"].values
features_forest = train[["Pclass", "Age", "Sex","Fare"]].values
test_features = test[["Pclass", "Age", "Sex","Fare"]].values

forest = tree.DecisionTreeClassifier(max_depth =10, min_samples_split = 5, random_state = 1)
my_forest = forest.fit(features_forest, target)


pred_forest = my_forest.predict(test_features)


PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
