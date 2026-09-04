import numpy as np
import math

class KMeansEngine:
    """
    [TV2] K-Means Clustering Engine
    Calculates iteration-by-iteration distance matrices D_t, partition matrices U_t,
    centroid updates, and returns 2D plot coordinates for Chart.js.
    """

    @staticmethod
    def run_kmeans(points, k=2, initial_centroids=None, max_iter=10):
        """
        points: list of dicts [{'id': 'P1', 'x': 1.0, 'y': 2.0}, ...] or 2D coordinates
        k: int (number of clusters)
        initial_centroids: list of dicts or list of [x, y] coordinates
        """
        # Format points as numpy array
        point_labels = []
        coords = []
        for i, p in enumerate(points):
            label = p.get('id', f"P{i+1}")
            point_labels.append(label)
            coords.append([float(p['x']), float(p['y'])])

        X = np.array(coords)
        n_samples = len(X)

        if initial_centroids and len(initial_centroids) == k:
            centroids = np.array([[float(c['x']), float(c['y'])] for c in initial_centroids])
        else:
            # Pick first K points as default initial centroids
            centroids = X[:k].copy()

        iterations = []
        converged = False
        t = 0

        while not converged and t < max_iter:
            t += 1
            # 1. Compute Distance Matrix D^(t) (shape: n_samples x k)
            dist_matrix = np.zeros((n_samples, k))
            for i in range(n_samples):
                for j in range(k):
                    dist = np.sqrt(np.sum((X[i] - centroids[j]) ** 2))
                    dist_matrix[i, j] = round(float(dist), 4)

            # 2. Compute Partition Matrix U^(t) (shape: n_samples x k)
            partition_matrix = np.zeros((n_samples, k), dtype=int)
            cluster_assignments = np.argmin(dist_matrix, axis=1)

            for i in range(n_samples):
                partition_matrix[i, cluster_assignments[i]] = 1

            # 3. Save iteration state
            iter_data = {
                "iteration": t,
                "centroids": centroids.round(4).tolist(),
                "distance_matrix_D": dist_matrix.tolist(),
                "partition_matrix_U": partition_matrix.tolist(),
                "cluster_assignments": cluster_assignments.tolist(),
                "point_labels": point_labels,
                "points_x": X[:, 0].tolist(),
                "points_y": X[:, 1].tolist(),
            }
            iterations.append(iter_data)

            # 4. Compute New Centroids
            new_centroids = np.zeros((k, 2))
            for j in range(k):
                cluster_points = X[cluster_assignments == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = cluster_points.mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            # Check convergence
            if np.allclose(centroids, new_centroids, atol=1e-4):
                converged = True

            centroids = new_centroids

        return {
            "k": k,
            "total_iterations": len(iterations),
            "converged": converged,
            "point_labels": point_labels,
            "iterations": iterations,
            "final_centroids": centroids.round(4).tolist()
        }
