import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

penguins_df = pd.read_csv("penguins.csv")
print(penguins_df.head())
penguins_df.dropna(inplace=True)
output = penguins_df['species']
features = penguins_df[['island', 'bill_length_mm', 'bill_depth_mm', 
                        'flipper_length_mm', 'body_mass_g', 'sex']]
# one-hot-encoding (or creating dummy variables for our text columns)
features = pd.get_dummies(features)
print("Here are our output variables")
print(output.head())
print("Here are our feature variables")
print(features.head())

output, uniques = pd.factorize(output)

X_train, X_test, y_train, y_test = train_test_split(features, output,
                                                    test_size=0.8)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test.values)
score = accuracy_score(y_true=y_test, y_pred=y_pred)
print("our accuracy score is {}".format(score))

rf_pickle = open("random_forest_penguin.pickle", 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

output_pickle = open("output_penguin.pickle", 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()

fig, ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_, y=features.columns)
plt.title("Which features are the most important for species prediction?")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.tight_layout()
fig.savefig("feature_importance.png")