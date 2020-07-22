import joblib
import pandas as pd

model_name = 'my_model_for_healthcare'
# model_name = 'my_model_for_dps'
dt = joblib.load(model_name)
df = pd.read_csv('training.csv')
x = df.drop(columns='prognosis', axis=1)
a = list(range(2, 134))
for i in range(len(x.columns)):
    print(str(i + 1 + 1) + ":", x.columns[i])
choices = input('Enter the Serial no.s which is your Symptoms are exist:  ')
b = [int(x) for x in choices.split()]
print(f'input b = {b}')
count = 0
while count < len(b):
    item_to_replace = b[count]
    replacement_value = 1
    indices_to_replace = [i for i, x in enumerate(a) if x == item_to_replace]
    count += 1
    for i in indices_to_replace:
        a[i] = replacement_value
a = [0 if x != 1 else x for x in a]
a = [0 if x != 1 else x for x in a]
print(f'final prediction list of symptoms = {a}')
y_diagnosis = dt.predict([a])
y_pred_2 = dt.predict_proba([a])
print(('Name of the infection = %s \nConfidence score of : = %s') % (y_diagnosis[0], y_pred_2.max() * 100), '%')

print(".\n.\n.\n")
