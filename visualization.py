import matplotlib.pyplot as plt
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd


def elbow(array):
    distortions = []
    K_range = range(1, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(array)
        distortions.append(km.inertia_)

    plt.plot(K_range, distortions, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (Within-cluster variance)")
    plt.title("Elbow Method for Optimal k")
    plt.show()

def Two_D_plot(array,labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(array)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20')
    plt.title('Kmeans Clustering Results (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label='Cluster')
    plt.show()
def three_D_plot(array,labels):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(array)


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(pca_result[:, 0],
                        pca_result[:, 1],
                        pca_result[:, 2],
                        c=labels,
                        cmap='tab20',
                        alpha=0.6)

    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    plt.title('3D PCA of Text Data')

    plt.colorbar(scatter)
    ax.grid(True)
    ax.view_init(30, 45)
    plt.show()
def three_D_interactive(array,labels):
    # 2. Interactive Plotly Plot
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(array)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Cluster'] = labels

    fig = px.scatter_3d(pca_df,
                        x='PC1',
                        y='PC2',
                        z='PC3',
                        color='Cluster',
                        color_continuous_scale='sunsetdark',
                        title='Interactive 3D PCA Plot')

    fig.update_layout(
        scene = dict(
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            zaxis_title='Third Principal Component'
        ),
        width=900,
        height=700,
    )

    fig.update_traces(marker=dict(size=5,
                                opacity=0.8))
    fig.show()

