# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "scikit-learn==1.8.0",
#     "scipy==1.17.1",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.linalg import eigh
    import sklearn
    from sklearn.neighbors import NearestNeighbors


@app.cell
def _():
    n_nodes = 100
    return (n_nodes,)


@app.cell
def _(n_nodes):
    moons, _ = sklearn.datasets.make_moons(n_samples=n_nodes)
    return (moons,)


@app.function
def scatter(X):
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.axis("equal")
    return plt.gca()


@app.cell
def _(moons):
    scatter(moons)
    return


@app.cell
def _(moons):
    neighbors = NearestNeighbors(n_neighbors=2).fit(moons)
    A = neighbors.kneighbors_graph(moons).toarray()
    adjacency_matrix = np.maximum(A, A.T)
    adjacency_matrix
    return (adjacency_matrix,)


@app.cell(hide_code=True)
def _(adjacency_matrix, moons, n_nodes):
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency_matrix[i, j]:
                plt.plot(
                    [moons[i, 0], moons[j, 0]],
                    [moons[i, 1], moons[j, 1]],
                    "-",
                    color="grey",
                    lw=0.5,
                )
    scatter(moons)
    return


@app.cell
def _(adjacency_matrix):
    plt.imshow(adjacency_matrix, cmap="grey")
    plt.axis("off")
    plt.gca()
    return


@app.cell
def _(moons):
    color_by_clusters(moons, compute_clusters(moons))
    return


@app.cell
def _(adjacency_matrix):
    L = laplacian_matrix(adjacency_matrix)
    return (L,)


@app.cell
def _(L):
    eigenvalues, eigenvectors = eigh(L)
    spectral_embedding = eigenvectors[:, 1:3]
    color_by_clusters(spectral_embedding, compute_clusters(spectral_embedding))
    return (spectral_embedding,)


@app.cell
def _(spectral_embedding):
    spectral_embedding.shape
    return


@app.cell
def _(moons, spectral_embedding):
    color_by_clusters(moons, compute_clusters(spectral_embedding[:, :2]))
    return


@app.function
def laplacian_matrix(adjacency_matrix):
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    L = degree_matrix - adjacency_matrix
    return L


@app.function
def color_by_clusters(X, cluster_labels):
    plt.scatter(X[:, 0], X[:, 1], s=1, c=cluster_labels, cmap='viridis')
    plt.axis("equal")
    return plt.gca()


@app.function
def compute_clusters(X):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0)
    return kmeans.fit_predict(X)


if __name__ == "__main__":
    app.run()
