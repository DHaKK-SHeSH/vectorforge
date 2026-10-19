"""k-means clustering from scratch.  <-- YOU write the four TODO functions.

IVF (next file) needs k-means to split the database into clusters, so this
comes first. Goal: make tests/test_kmeans.py pass.

The algorithm (Lloyd's), on n points x_1..x_n and k centroids c_1..c_k:

    repeat:
        1. ASSIGN  each point to its nearest centroid
        2. UPDATE  each centroid to the mean of the points assigned to it

The quantity it minimises is the INERTIA (sum of squared errors):

    inertia = sum_i || x_i - c_{label(i)} ||^2

Step 1 can only lower it (each point picks its best centroid), and step 2 can
only lower it (the mean is the point with the smallest summed squared distance
to a set of points). So inertia never goes up. A test checks exactly that.

Suggested order: assign -> update_centroids -> kmeans (with random init) ->
run the tests -> init_kmeanspp -> run the tests again.
"""

from typing import List, Tuple

import numpy as np


def init_random(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick k distinct data points as the starting centroids. (Given, as an example.)"""
    idx = rng.choice(len(x), size=k, replace=False)
    return x[idx].copy()


def init_kmeanspp(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialisation. Returns (k, d) float32 centroids, each one a data point.

    1. Pick the first centroid uniformly at random from x.
    2. For every point, D(x) = squared distance to its NEAREST centroid chosen so far.
    3. Pick the next centroid at random, with probability proportional to D(x).
       (Far-away points are likely; points already covered are unlikely.)
    4. Repeat 2-3 until there are k centroids.

    Hints:
    - rng.choice(n, p=probabilities) samples one index with the given probabilities
      (probabilities must sum to 1: divide D by D.sum()).
    - You don't need to recompute D from scratch each round: after adding a new
      centroid c, D = np.minimum(D, squared distance of every point to c).
    """
    n = len(x)
    first = rng.integers(n)
    chosen = [x[first]]
    D = ((x-x[first])**2).sum(axis=1)
    for _ in range(k-1):
         p = D / D.sum()
         i = rng.choice(n, p = p)
         chosen.append(x[i])
         D = np.minimum(D,((x-x[i])**2).sum(axis=1))
    return np.array(chosen,dtype = np.float32)
         


def assign(x: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, sq_dists), both shape (n,).

    labels[i]   = index of the centroid nearest to x[i]
    sq_dists[i] = squared L2 distance from x[i] to that centroid

    Hint: this is your flat.py distance matrix again, with the centroids playing
    the role of the database. Then argmin along each row. No Python loop.
    """
    x = np.asarray(x,dtype=np.float32)
    centroids = np.asarray(centroids,dtype=np.float32)
    
    x_norms = (x**2).sum(axis=1)
    centroids_norms = (centroids**2).sum(axis=1)
    D = x_norms[:,None] - 2 * (x @ centroids.T) + centroids_norms
    D = np.maximum(D,0)
    labels = D.argmin(axis=1)
    sq_dist = D.min(axis=1)
    return labels,sq_dist


def update_centroids(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """Return new (k, d) centroids: centroid j = mean of the points with label j.

    EMPTY CLUSTERS: if no point has label j, its mean is undefined (0/0 = NaN).
    Don't leave it there. Re-seed it to a random data point instead.

    Hints:
    - np.bincount(labels, minlength=k) counts points per cluster.
    - np.add.at(sums, labels, x) adds each row of x into sums[label] — a loop-free
      "group by and sum". Then divide by the counts.
    - Divide only where count > 0, and handle the rest separately.
    """
    k = len(centroids)
    counts = np.bincount(labels,minlength=k)
    sums = np.zeros((k,x.shape[1]))
    np.add.at(sums,labels,x)
    new = centroids.copy()
    nonempty = counts > 0
    new[nonempty] = sums[nonempty]/counts[nonempty][:,None]
    for j in np.flatnonzero(~nonempty):
        new[j] = x[rng.integers(len(x))]
    return new
    


def kmeans(x: np.ndarray, k: int, n_iter: int = 20, init: str = "kmeans++",
           seed: int = 0, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Run Lloyd's algorithm. Returns (centroids (k, d), labels (n,), inertia_history).

    inertia_history[t] = inertia after the assign step of iteration t.

    Each iteration:
        1. labels, sq_dists = assign(...)
        2. append the inertia (sum of sq_dists) to the history
        3. stop early if the inertia improved by less than `tol` (relative)
           compared with the previous iteration
        4. centroids = update_centroids(...)
    """
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if init == "random":
        centroids = init_random(x, k, rng)
    elif init == "kmeans++":
        centroids = init_kmeanspp(x, k, rng)
    else:
        raise ValueError("init must be 'random' or 'kmeans++'")

    history = []
    for t in range(n_iter):
        labels,sq_dist = assign (x,centroids)
        history.append(float(sq_dist.sum()))
        if len(history) >1 and history[-2]-history[-1] <= tol:
            break
        centroids = update_centroids(x, labels,centroids,rng)
        
    return centroids, labels, history
