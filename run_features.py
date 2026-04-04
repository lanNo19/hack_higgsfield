from src.features.build_features import build_feature_matrix

X, y = build_feature_matrix('train', force_rebuild=True)
print(X.shape)
print(X.head())
print(y.value_counts())
