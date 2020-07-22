import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurations
sns.set(style="darkgrid")
df = pd.read_csv('training.csv')

# Displaying Database Details
print(f'Dataset Details :: ')
print(f'Total Rows --> {df.shape[0]}')
print(f'Total Columns --> {df.shape[1]}')
print(f'Total Symptoms --> {df.shape[1]-1}')
print(f"Total Diseases --> {df['prognosis'].nunique()}")
print(f'Symptoms Data types --> {df.drop("prognosis", axis=1).dtypes.unique()}')
print(f'Disease Data types --> {df["prognosis"].dtypes}')

# Dsipaly symptoms
print('\nList of Symptoms :: ')
symptoms = df.drop(['prognosis'], axis=1).columns.tolist()
print('\n'.join([f'{index+1} : {symptom}' for index, symptom in enumerate(symptoms)]))

# Display disease
print('\nList of Diseases :: ')
diseases = df['prognosis'].unique().tolist()
print('\n'.join([f'{index+1} : {disease}' for index, disease in enumerate(diseases)]))

sns.set(style='darkgrid', color_codes=True)
# TODO :: Bar plots
i = 0
for times in range(11):
    fig, axs = plt.subplots(2, 6, sharex='all', sharey='all')
    r = 0
    c = 0
    for r in range(2):
        for c in range(6):
            if i < df.shape[1]-1:
                sns.countplot(x=df[df.columns[i]], data=df, ax=axs[r][c])
                i += 1
                c += 1
        r += 1
    #plt.savefig(f'bar_plots/{i-12+1} - {i}')
    plt.show()

# TODO :: Symptom Disease relationship plots
# for symptom in symptoms:
#     print(f'plotting for --> {symptom}')
#     sns_plot = sns.barplot(x=symptom, y='prognosis', data=df)
#     fig = sns_plot.get_figure()
#     fig.savefig(f'symptom_disease_relationship_plots/{symptom}')
#     plt.close(fig)
# sns.countplot(x='nodal_skin_eruptions', data=df)
# sns.countplot(x='continuous_sneezing', data=df)

#sns.factorplot('itching', data=df, kind='count')
#sns.relplot(x='itching', y='prognosis', data=df, hue='itching')
