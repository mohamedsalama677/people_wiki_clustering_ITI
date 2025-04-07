import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer
from preprocessing import preprocess_text_simple, tfidf_scaled
from visualization import Two_D_plot,three_D_plot,three_D_interactive
df = pd.read_csv('Dataset\people_wiki.csv')

pipeline = Pipeline([
    ('preprocess', FunctionTransformer(lambda X: [preprocess_text_simple(text) for text in X], validate=False)),
    ('tfidf', FunctionTransformer(lambda X: tfidf_scaled(pd.Series(X)), validate=False)),
    ('kmeans', KMeans(n_clusters=3))
])


preprocessed_text = pipeline.named_steps['preprocess'].transform(df['text'])

tfidf_features = pipeline.named_steps['tfidf'].transform(preprocessed_text)

kmeans = pipeline.named_steps['kmeans']
labels = kmeans.fit_predict(tfidf_features)

Two_D_plot(tfidf_features,labels)
three_D_plot(tfidf_features,labels)
three_D_interactive(tfidf_features,labels)