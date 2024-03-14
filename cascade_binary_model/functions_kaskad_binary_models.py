import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from psi import calculate_psi
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import seaborn as sns
from collections import Counter
import numpy as np





def custom_agg(x, target):
    clients = x['id'].nunique()
    gini_val = roc_auc_score(x[target], x['proba'])
    return pd.Series({'clients': clients, 'roc_auc': gini_val})


list_product = ['α', 'β', 'γ', 'δ', 'ε', 'η', 'θ', 'λ', 'μ', 'π', 'ψ', 'other']

end_segments = start_segments = ['{α}',
 '{}',
 '{other}',
 '{α, η}',
 '{α, γ}',
 '{α, β}',
 '{α, δ}',
 '{α, ε}',
 '{α, θ}',
 '{α, ψ}',
 '{α, μ}',
 '{α, ε, η}',
 '{α, ε, θ}',
 '{α, λ}',
 '{α, ε, ψ}',
 '{λ}',
 '{α, π}']

#### churn

def filter_and_check(row, product):
    return product in row['start_cluster']

def check_churn(row, product):
    if product in row['start_cluster'] and product not in row['end_cluster']:
        return 1
    else:
        return 0

def plot_importance(model):  
    importances = model.get_feature_importance()

    feature_names = model.feature_names_
    feature_importances = pd.DataFrame({'feature_names': feature_names, 'feature_importance': importances})

    feature_importances.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10,8))
    sns.barplot(x='feature_importance', y='feature_names', data=feature_importances[:25])
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature Names')
    plt.show()

def binary_model(train_df, product, target, features, cat_cols):
    
    train_df_churn = train_df[train_df.apply(filter_and_check, axis=1, product = product)]
    train_df_churn[target] = train_df_churn.apply(check_churn, axis=1, product = product)
    
    print(train_df_churn.groupby('date')[target].mean())
    
    
#     X_train = train_df_churn[train_df_churn['date'] != 'month_2'][features]
#     y_train = train_df_churn[train_df_churn['date'] != 'month_2'][target]


#     X_val = train_df_churn[train_df_churn['date'] == 'month_2'][features]
#     y_val = train_df_churn[train_df_churn['date'] == 'month_2'][target]



    y = train_df_churn[target]
    X = train_df_churn[features]


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=55)
    
    mean_target = train_df_churn[target].mean()


    model_churn = CatBoostClassifier(iterations=800, 
                     learning_rate=0.1, 
                     depth=6, 
                     l2_leaf_reg=3, 
                     loss_function='Logloss',
                      eval_metric='AUC',
                      cat_features=cat_cols)
        
    model_churn.fit(X_train, y_train, eval_set=(X_val, y_val), verbose = 100)
        
    train_df_churn['proba'] = model_churn.predict_proba(train_df_churn[features])[:,1]
    
    grouped_df = train_df_churn.groupby('date').apply(custom_agg, target=target).reset_index()
        
    return model_churn, grouped_df




#### propensity



def classes_weights(y):
    class_counts = Counter(y)
    total_samples = len(y)

    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    return class_weights

def filter_and_check_propensity(row, product):
    return product not in row['start_cluster']

def check_propensity(row, product):
    if product not in row['start_cluster'] and product in row['end_cluster']:
        return 1
    else:
        return 0

def binary_model_propensity(train_df, product, target, features, cat_cols):
    
    train_df_propensity = train_df[train_df.apply(filter_and_check_propensity, axis=1, product = product)]
    train_df_propensity[target] = train_df_propensity.apply(check_propensity, 
    axis=1, product = product)
    
    print(train_df_propensity.groupby('date')[target].mean())

    y = train_df_propensity[target]
    X = train_df_propensity[features]


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=666)
    
    mean_target = train_df_propensity[target].mean()
    print(mean_target)
    

    
    model_propensity = CatBoostClassifier(iterations=800, 
                     learning_rate=0.1, 
                     depth=6, 
                     l2_leaf_reg=3, 
                     loss_function='Logloss',
                      eval_metric='AUC',
                      cat_features=cat_cols)

    model_propensity.fit(X_train, y_train, eval_set=(X_val, y_val), verbose = 100)
        
    train_df_propensity['proba'] = model_propensity.predict_proba(train_df_propensity[features])[:,1]
    
    grouped_df = train_df_propensity.groupby('date').apply(custom_agg, target=target).reset_index()
        
    return model_propensity, grouped_df






