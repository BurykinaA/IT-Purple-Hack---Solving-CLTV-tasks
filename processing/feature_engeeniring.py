import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clever_one_hot(df, col):  # переписать в lable encoding?
    top_4 = df[col].value_counts().index[:4]
    df.loc[~df[col].isin(top_4), col] = "other"
    one_hot_encoded = pd.get_dummies(df[col], prefix=col)
    df = df.drop(col, axis=1)
    return pd.concat([df, one_hot_encoded], axis=1)


def read_csv(path_test: str, path_train: str):
    train_df = pd.read_parquet(path_train)
    test_df = pd.read_parquet(path_test)
    return train_df, test_df


def feature_prossesing_Lable_encoder(df, type: str["test", "train"]):
    one_hot_encoded = pd.get_dummies(df["segment"], prefix="seg")
    df = df.drop("segment", axis=1)
    df = pd.concat([df, one_hot_encoded], axis=1)

    df["ogrn_year"] = df["ogrn_year"].fillna("_-1")
    df["ogrn_month"] = df["ogrn_month"].fillna("_-1")
    df["ogrn_year"] = df["ogrn_year"].apply(lambda x: int(x.split("_")[-1]))
    df["ogrn_month"] = df["ogrn_month"].apply(lambda x: int(x.split("_")[-1]))
    df["lasting"] = df["ogrn_year"] * 12 + df["ogrn_month"]
    df["lasting"] = df["lasting"].apply(lambda x: x if x >= 0 else -1)

    del df["ogrn_year"]
    del df["ogrn_month"]

    for i in ["channel_code", "city_type"]:
        df = clever_one_hot(df, i)

    label_encoder = LabelEncoder()
    df["start_cluster"] = label_encoder.fit_transform(df["start_cluster"])
    if type == "train":
        df["end_cluster"] = label_encoder.transform(df["end_cluster"])
    category_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )

    df = df.drop(df.select_dtypes(include=["object"]).columns, axis=1)
    df = df.fillna(-1)

    return df, category_mapping


def feature_prossesing_One_hot(df):
    one_hot_encoded = pd.get_dummies(df["segment"], prefix="seg")
    df = df.drop("segment", axis=1)
    df = pd.concat([df, one_hot_encoded], axis=1)

    df["ogrn_year"] = df["ogrn_year"].fillna("_-1")
    df["ogrn_month"] = df["ogrn_month"].fillna("_-1")
    df["ogrn_year"] = df["ogrn_year"].apply(lambda x: int(x.split("_")[-1]))
    df["ogrn_month"] = df["ogrn_month"].apply(lambda x: int(x.split("_")[-1]))
    df["lasting"] = df["ogrn_year"] * 12 + df["ogrn_month"]
    df["lasting"] = df["lasting"].apply(lambda x: x if x >= 0 else -1)

    del df["ogrn_year"]
    del df["ogrn_month"]

    for i in ["channel_code", "city_type"]:
        df = clever_one_hot(df, i)

    label_encoder = LabelEncoder()
    df["start_cluster"] = label_encoder.fit_transform(df["start_cluster"])
    df["end_cluster"] = label_encoder.transform(df["end_cluster"])
    category_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )

    df = df.drop(df.select_dtypes(include=["object"]).columns, axis=1)
    df = df.fillna(-1)

    return df, category_mapping


def get_all_client_data(df, type: str["test", "train"]):
    grouped_df_first = df.groupby("id").first().reset_index()
    merged_df = pd.merge(df, grouped_df_first, on="id", suffixes=("", "_first"))

    grouped_df_second = test_df.groupby("id").nth(1).reset_index()
    merged_df = pd.merge(
        merged_df, grouped_df_second, on="id", suffixes=("", "_second")
    )

    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    if type == "test":
        return merged_df[merged_df["date"] == "month_6"]
    else:
        return merged_df[merged_df["date"] == "month_3"]


def correlation_filter(train_df, test_df):
    corr_matrix = train_df.corr().abs()

    # Получение верхнего треугольника матрицы корреляции (без диагонали)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    )

    # Нахождение колонок, где корреляция больше 0.9
    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > 0.95)
    ]

    test_df = test_df.drop(to_drop, axis=1)
    train_df = train_df.drop(to_drop, axis=1)

    return test_df, train_df
