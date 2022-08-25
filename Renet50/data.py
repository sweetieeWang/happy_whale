import pandas as pd
df = pd.read_csv('../train.csv')
df

# df_mini = df.sample(n=1230, frac=None, replace=False, weights=None, random_state=None, axis=None)

condition = False
k = 1000000000
while(k > 0 and condition == False):
    print(k)
    df_mini = df.sample(n=1230, frac=None, replace=False, weights=None, random_state=None, axis=None)
    condition = df_mini['individual_id'].value_counts().shape[0] == 1000
    k -= 1
    print(k)

print(df_mini)
print(df_mini.describe())
# df_mini.to_csv("train_1000.csv")


print(df_mini.describe())
df_mini.to_csv("train_1000.csv", index = False)
