import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('training.csv')
print('Before shuffling')
print(df.head())
df = df.sample(frac=1).reset_index(drop=True)
# print('After shuffling')
# print(df.head())

kmax = 11
dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=10)

test_scores = {}
train_scores = {}
for i in range(2, kmax, 2):
    kf = KFold(n_splits=i)
    sum_train = 0
    sum_test = 0
    max_accuracy = 0.0
    for train_index, test_index in kf.split(df):
        train_data = df.iloc[train_index, :]
        test_data = df.iloc[test_index, :]
        x_train = train_data.drop(["prognosis"], axis=1)
        y_train = train_data["prognosis"]
        x_test = test_data.drop(["prognosis"], axis=1)
        y_test = test_data["prognosis"]
        algo_model = dt.fit(x_train, y_train)
        sum_train += algo_model.score(x_train, y_train)
        y_pred = algo_model.predict(x_test)
        sum_test += accuracy_score(y_test, y_pred)
        if (accuracy_score(y_test, y_pred) < 1.0) and (max_accuracy < accuracy_score(y_test, y_pred)):
            max_accuracy = accuracy_score(y_test, y_pred)
    average_test = sum_test / i
    average_train = sum_train / i
    test_scores[i] = average_test
    train_scores[i] = average_train
    print(f'K value : {i}')

print(f'Train Score = {train_scores}\nTest Score = {test_scores}')
k_fold_value_final = 0
max_score = float(0.0)
for k in test_scores.keys():
    if (test_scores[k] < 1.0) and (max_score < test_scores[k]):
        k_fold_value_final = k
        max_score = test_scores[k]
print(f'Final K Value = {k_fold_value_final} with score = {max_score}')

values = sorted(test_scores.items())
x, y = zip(*values)
plt.plot(x, y)

test_scores = {}
train_scores = {}
for i in range(2, k_fold_value_final + 1, 2):
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

# plt.show()
# plt.grid()
joblib.dump(dt, 'dt_model')

# prediction
# dt = joblib.load('dt_model')
# x = df.drop(columns='prognosis', axis=1)
# y = df['prognosis']
# a = list(range(1, 133))
# i_name = (input('Enter your name :'))
# i_age = (int(input('Enter your age:')))
# for i in range(len(x.columns)):
#     print(str(i + 1 + 1) + ":", x.columns[i])
# choices = input('Enter the Serial no.s which is your Symptoms are exist:  ')
# b = [int(x) for x in choices.split()]
# count = 0
# while count < len(b):
#     item_to_replace = b[count]
#     replacement_value = 1
#     indices_to_replace = [i for i, x in enumerate(a) if x == item_to_replace]
#     count += 1
#     for i in indices_to_replace:
#         a[i] = replacement_value
#         print(f'replaced value = {a[i]}')
# a = [0 if x != 1 else x for x in a]
# print(f'final prediction list of symptoms = {a}')
# y_diagnosis = dt.predict([a])
# y_pred_2 = dt.predict_proba([a])
# print(('Name of the infection = %s , confidence score of : = %s') % (y_diagnosis[0], y_pred_2.max() * 100), '%')
# print(('Name = %s , Age : = %s') % (i_name, i_age))
#
# print(f'y_diagnosis = {y_diagnosis}')
# print(f'y_pred_2 = {len(y_pred_2)}')

print(".\n.\n.\n")

# test 1 = 1 2 3 103 - Fungal Infection
# test 2 = 4 5 6 104 - allergy
