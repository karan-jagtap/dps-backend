import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


def evaluate(df_c, kmax, algo):
    test_scores = {}
    train_scores = {}
    for i in range(2, kmax, 2):
        kf = KFold(n_splits=i)
        sum_train = 0
        sum_test = 0
        data = df_c
        for train, test in kf.split(data):
            train_data = data.iloc[train, :]
            test_data = data.iloc[test, :]
            x_train = train_data.drop(["prognosis"], axis=1)
            y_train = train_data["prognosis"]
            x_test = test_data.drop(["prognosis"], axis=1)
            y_test = test_data["prognosis"]
            algo_model = algo.fit(x_train, y_train)
            sum_train += algo_model.score(x_train, y_train)
            y_pred = algo_model.predict(x_test)
            sum_test += accuracy_score(y_test, y_pred)
        average_test = sum_test / i
        average_train = sum_train / i
        test_scores[i] = average_test
        train_scores[i] = average_train
        print(f'K value : {i}')
    return train_scores, test_scores


# print('importing training dataset...')
df = pd.read_csv('training.csv')
df = df.sample(frac=1)
# print(df.head())
#
# rows = df.shape[0]
# cols = df.shape[1]
# print(f"\nSize of training dataset : \nRows : {rows}\nCols : {cols}")

# getting the count of null values in the whole dataset in descending order
# print(df.isnull().sum().sort_values(ascending=False))

# col_names = df.columns
# print(f"\nColumns List : {col_names}")

disease_percentage = df['prognosis'].value_counts(normalize=True)
print(f"\nEach disease percent wise : \n{disease_percentage}")
print(type(disease_percentage))
disease_percentage.plot.bar()
plt.subplots_adjust(bottom=0.3)
# TODO :: plt.show()

print(f"Series Data types : {df.dtypes.unique()}")

# TODO ::
# for x in range(df.shape[1]):
#     sns.countplot(df[df.columns[x]])
#     plt.show()

x = df.drop(columns='prognosis', axis=1)
y = df['prognosis']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#
# mnb = MultinomialNB()
# mnb = mnb.fit(x_train, y_train)
#
# y_pred = mnb.predict(x_test)
# mnb_accuracy = accuracy_score(y_pred, y_test)
# print(f"Naive Bayes Accuracy : {mnb_accuracy}")
#
# mnb_scores = cross_val_score(mnb, x_test, y_test, cv=3)
# mnb_mean_score = mnb_scores.mean()
# print(f'Cross Validation Mean Score {mnb_mean_score}')
#
# real_diseases = y_test.values
# for pred, actual in zip(y_pred, real_diseases):
#     if pred == actual:
#         print(f'Predicted : {pred} --> Actual : {actual}')
#     else:
#         print(f'Wrong Prediction :: \nPredicted : {pred} --> Actual : {actual}')

gbm = GradientBoostingClassifier()
log = LogisticRegression()
dt = DecisionTreeClassifier()
ran = RandomForestClassifier()
mnb = MultinomialNB()
algo_dict = {
    'l_o_g': log,
    'd_t': dt,
    'r_a_n': ran,
    'g_b_m': gbm,
    'm_n_b': mnb,
}

max_kfold = 11
algo_train_scores = {}
algo_test_scores = {}
for algo_name in algo_dict.keys():
    print(algo_name)
    tr_score, tst_score = evaluate(df, max_kfold, algo_dict[algo_name])
    algo_train_scores[algo_name] = tr_score
    algo_test_scores[algo_name] = tst_score
print(algo_train_scores)
print(algo_test_scores)

df_test = pd.DataFrame(algo_test_scores)
df_train = pd.DataFrame(algo_train_scores)

df_test.plot(grid=1)
plt.show()
plt.grid()

test_scores = {}
train_scores = {}
for i in range(2, 4, 2):
    print(f'iteration {i}')
    kf = KFold(n_splits=i)
    sum_train = 0
    sum_test = 0
    data = df
    for train, test in kf.split(data):
        train_data = data.iloc[train, :]
        test_data = data.iloc[test, :]
        x_train = train_data.drop(["prognosis"], axis=1)
        y_train = train_data['prognosis']
        x_test = test_data.drop(["prognosis"], axis=1)
        y_test = test_data["prognosis"]
        dt = dt.fit(x_train, y_train)
        sum_train += dt.score(x_train, y_train)
        y_pred = dt.predict(x_test)
        sum_test += accuracy_score(y_test, y_pred)
    average_test = sum_test / i
    average_train = sum_train / i
    test_scores[i] = average_test
    train_scores[i] = average_train
    print("kvalue: ", i)
print(f'Final Train Score - {train_scores}')
print(f'Final Test Score - {test_scores}')

joblib.dump(dt, 'my_model_for_dps')

# Optional
a = list(range(2, 134))
i_name = (input('Enter your name :'))
i_age = (int(input('Enter your age:')))
for i in range(len(x.columns)):
    print(str(i + 1 + 1) + ":", x.columns[i])
choices = input('Enter the Serial no.s which is your Symptoms are exist:  ')
b = [int(x) for x in choices.split()]
count = 0
while count < len(b):
    item_to_replace = b[count]
    replacement_value = 1
    indices_to_replace = [i for i, x in enumerate(a) if x == item_to_replace]
    count += 1
    for i in indices_to_replace:
        a[i] = replacement_value
        print(f'replaced value = {a[i]}')
a = [0 if x != 1 else x for x in a]
print(f'final prediction list of symptoms = {a}')
y_diagnosis = dt.predict([a])
y_pred_2 = dt.predict_proba([a])
print(('Name of the infection = %s , confidence score of : = %s') % (y_diagnosis[0], y_pred_2.max() * 100), '%')
print(('Name = %s , Age : = %s') % (i_name, i_age))

print(".\n.\n.\n")
