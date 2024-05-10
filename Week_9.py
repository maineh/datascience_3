import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import Stemmer
import time

start_time = time.time()
stemmer = Stemmer.Stemmer('english')

# Load data
df_train = pd.read_csv('input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('input/test.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('input/attributes.csv')
df_pro_desc = pd.read_csv('input/product_descriptions.csv')

num_train = df_train.shape[0]

# Preprocess and feature extraction
def str_stemmer(s):
    return " ".join(stemmer.stemWords(s.lower().split()))

def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())

def jaccard_similarity(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')

# Stemming
df_all['search_term'] = df_all['search_term'].apply(str_stemmer)
df_all['product_title'] = df_all['product_title'].apply(str_stemmer)
df_all['product_description'] = df_all['product_description'].apply(str_stemmer)
df_all['name'] = df_all['name'].fillna('').apply(str_stemmer)
df_all['value'] = df_all['value'].fillna('').apply(str_stemmer)

# Features
features = ['len_of_query', 'jaccard_title', 'jaccard_description', 'word_in_title', 'word_in_description', 'word_in_name', 'word_in_value']
df_all['len_of_query'] = df_all['search_term'].apply(lambda x: len(x.split()))

# Jaccard Similarity
df_all['jaccard_title'] = df_all.apply(lambda x: jaccard_similarity(x['search_term'], x['product_title']), axis=1)
df_all['jaccard_description'] = df_all.apply(lambda x: jaccard_similarity(x['search_term'], x['product_description']), axis=1)
df_all['word_in_title'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['product_title']), axis=1)
df_all['word_in_description'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['product_description']), axis=1)
df_all['word_in_name'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['name']), axis=1)
df_all['word_in_value'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['value']), axis=1)

df_all = df_all.drop(['search_term','product_title','product_description', 'name', 'value'],axis=1)

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train RandomForestRegressor
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
rf.fit(X_train, y_train)

# Extract feature importances and sort them
importances = rf.feature_importances_
sorted_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

# Output the top 5 important features
print("Top 5 most important features:")
for feature, importance in sorted_features[:5]:
    print(f"{feature}: {importance}")

# Prediction and Evaluation
y_val_pred = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("RMSE on validation set:", rmse)
print("Process finished --- %s seconds ---" % (time.time() - start_time))