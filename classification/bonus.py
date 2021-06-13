import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_tree
from preparation import X, y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X.shape[2] * X.shape[3]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X.shape[2] * X.shape[3]))

boost = XGBClassifier(objective='multiclass:softmax', learning_rate=0.1,
                      max_depth=3, n_estimators=10)
boost.fit(X_train, y_train)
preds = boost.predict(X_test)
print(sum(preds == y_test) / len(y_test))

plt.cla()
plt.clf()
plot_tree(boost, num_trees=3)
plt.tight_layout()
# plt.savefig('../report/images/boosting.png', dpi=300)
# plt.show()
