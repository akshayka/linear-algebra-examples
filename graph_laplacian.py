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
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.linalg import eigh
    import sklearn
    from sklearn.neighbors import NearestNeighbors


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This notebook shows a special way to represent graphs as matrices, capturing
    important characteristics of the graph's connectivity.

    The Laplacian matrix of an undirected graph with degree matrix $D$ and adjacency matrix $A$
    is the symmetric postive semidefinite matrix

    $$
    L = D - A.
    $$

    Here, $D$ is a diagonal matrix in $\mathbf{R}^{n \times n}$ with $D_{i,i}$ equal
    to the degree of node $i$, and

    $$
    A_{ij} \in \mathbf{R}^{n \times n} =
    \begin{cases}
    1, & \mbox{$i$ is connected to $j$} \\
    0, & \mbox{otherwise}.
    \end{cases}
    $$

    This _Laplacian matrix_ tells us
    how scalar potentials associated with nodes encoded in a vector $x$ vary across the graph, satisfying

    $$
    x^T L x = \sum_{i, j} (x_i - x_j)^2.
    $$

    **Spectral properties.** $L$ has many interesting spectral properties. For example, an
    eigenvector with small eigenvalue means that connected nodes will have similar
    values (or "potentials") in that eigenvector, a property we can exploit for
    applications such as clustering and embedding.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Spectral clustering

    In this notebook we show how the eigenvectors of a graph Laplacian can be used to cluster
    data points in settings where a naive application of $k$ means fails. We will use data in the "moons" pattern.

    We start with a naive clustering.
    """)
    return


@app.cell
def _():
    moons, _ = sklearn.datasets.make_moons(n_samples=100)
    return (moons,)


@app.cell
def _(moons):
    scatter(moons)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $k$ means applied directly to this dataset fails to discover the two natural clusters.
    """)
    return


@app.cell
def _(moons):
    color_by_clusters(moons, compute_clusters(moons))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Neighborhood graph

    We can discover the natural clusters if we cluster the second eigenvector of a particular Laplacian matrix, interpreting the original data as a graph with two points connected if one is a nearest neighbor of the other, where the number of neighbors is a parameter.

    ### Adjacency matrix
    First, we form the graph's **adjacency matrix.**
    """)
    return


@app.cell
def _(moons):
    neighbors = NearestNeighbors().fit(moons)
    A = neighbors.kneighbors_graph(moons).toarray()
    adjacency_matrix = np.maximum(A, A.T)
    adjacency_matrix
    return (adjacency_matrix,)


@app.cell(hide_code=True)
def _(adjacency_matrix):
    plt.imshow(adjacency_matrix, cmap="grey")
    plt.axis("off")
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Laplacian matrix

    Next, we form the associated **Laplacian matrix** and compute its eigenvectors.
    """)
    return


@app.function
def laplacian_matrix(adjacency_matrix):
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    L = degree_matrix - adjacency_matrix
    return L


@app.cell
def _(adjacency_matrix):
    L = laplacian_matrix(adjacency_matrix)
    return (L,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The bottom eigenvector of $L$ is the all-ones eigenvector with eigenvalue $0$, which is uninteresting. The next eigenvector, however, also has a small eigenvalue, meaning its associated eigenvector has connected nodes placed near each other. This second eigenvector is known as the **Fiedler eigenvector.**
    """)
    return


@app.cell
def _(L):
    eigenvalues, eigenvectors = eigh(L)
    fiedler_eigenvector = eigenvectors[:, 1].reshape(-1, 1)
    return (fiedler_eigenvector,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can plot the entries of the Fiedler eigenvector. Notice how they sharply separate into two groups, suggesting that it may be useful in clustering the original data.
    """)
    return


@app.cell
def _(fiedler_eigenvector):
    plt.scatter(
        range(len(fiedler_eigenvector)),
        fiedler_eigenvector,
        c=compute_clusters(fiedler_eigenvector),
        s=10,
    )
    plt.xlabel("indices")
    plt.ylabel("value")
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Indeed, if we assign clusters based on a $k$-means clustering of the Fiedler eigenvector, we obtain the natural clustering on the moons dataset.
    """)
    return


@app.cell
def _(fiedler_eigenvector, moons):
    color_by_clusters(moons, compute_clusters(fiedler_eigenvector))
    return


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


@app.function
def scatter(X):
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.axis("equal")
    return plt.gca()


if __name__ == "__main__":
    app.run()
