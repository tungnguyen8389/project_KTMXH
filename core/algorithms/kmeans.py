import numpy as np
import math
import random

class KMeansEngine:
    """
    [TV2] K-Means Clustering Engine
    Calculates iteration-by-iteration distance matrices D_t, partition matrices U_t,
    centroid updates, WCSS (Within-Cluster Sum of Squares), and returns 2D plot coordinates
    along with rich cluster profiling & attrition insights for HR data.
    """

    @staticmethod
    def _init_centroids(X, k, method='first_k', seed=42):
        """
        Initialize k centroids using 'first_k', 'random', or 'kmeans++'.
        """
        n_samples = len(X)
        if method == 'random':
            rng = np.random.RandomState(seed)
            indices = rng.choice(n_samples, size=k, replace=False)
            return X[indices].copy()
        elif method == 'kmeans++':
            rng = np.random.RandomState(seed)
            # 1. Choose first center uniformly at random
            first_idx = rng.choice(n_samples)
            centers = [X[first_idx]]

            # 2. Choose remaining k-1 centers with probability proportional to D(x)^2
            for _ in range(1, k):
                dist_sq = np.array([
                    min(np.sum((x - c) ** 2) for c in centers)
                    for x in X
                ])
                total_dist = dist_sq.sum()
                if total_dist == 0:
                    probs = np.ones(n_samples) / n_samples
                else:
                    probs = dist_sq / total_dist
                next_idx = rng.choice(n_samples, p=probs)
                centers.append(X[next_idx])
            return np.array(centers)
        else:
            # Default: 'first_k' (take first k points)
            return X[:k].copy()

    @classmethod
    def run_kmeans(cls, points, k=2, initial_centroids=None, max_iter=15,
                   init_method='first_k', normalize=False, seed=42):
        """
        points: list of dicts [{'id': 'P1', 'x': 1.0, 'y': 2.0, 'meta': {...}}, ...]
        k: int (number of clusters)
        initial_centroids: list of dicts or list of [x, y] coordinates
        max_iter: int (maximum iterations)
        init_method: 'first_k' | 'random' | 'kmeans++'
        normalize: bool (scale features to [0, 1] during distance computation)
        """
        if not points:
            raise ValueError("Danh sách điểm (points) không được rỗng!")

        k = max(1, min(int(k), len(points)))
        max_iter = max(1, int(max_iter))

        # 1. Parse points and metadata
        point_labels = []
        coords_raw = []
        metadata_list = []

        for i, p in enumerate(points):
            label = p.get('id', f"P{i+1}")
            point_labels.append(str(label))
            coords_raw.append([float(p['x']), float(p['y'])])
            metadata_list.append(p.get('meta', {}))

        X_orig = np.array(coords_raw, dtype=float)
        n_samples = len(X_orig)

        # 2. Optional Feature Normalization (Min-Max)
        min_vals = X_orig.min(axis=0)
        max_vals = X_orig.max(axis=0)
        ranges = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)

        if normalize:
            X_calc = (X_orig - min_vals) / ranges
        else:
            X_calc = X_orig.copy()

        # 3. Centroid Initialization
        if initial_centroids and len(initial_centroids) == k:
            init_c = np.array([[float(c['x']), float(c['y'])] for c in initial_centroids])
            if normalize:
                centroids = (init_c - min_vals) / ranges
            else:
                centroids = init_c
        else:
            centroids = cls._init_centroids(X_calc, k, method=init_method, seed=seed)

        iterations = []
        converged = False
        t = 0

        while not converged and t < max_iter:
            t += 1

            # A. Compute Distance Matrix D^(t) (shape: n_samples x k)
            dist_matrix = np.zeros((n_samples, k))
            for i in range(n_samples):
                for j in range(k):
                    dist = np.sqrt(np.sum((X_calc[i] - centroids[j]) ** 2))
                    dist_matrix[i, j] = round(float(dist), 4)

            # B. Compute Partition Matrix U^(t) (shape: n_samples x k)
            partition_matrix = np.zeros((n_samples, k), dtype=int)
            cluster_assignments = np.argmin(dist_matrix, axis=1)

            for i in range(n_samples):
                partition_matrix[i, cluster_assignments[i]] = 1

            # C. Compute WCSS (Within-Cluster Sum of Squares / Inertia)
            wcss = 0.0
            for j in range(k):
                pts_in_cluster = X_calc[cluster_assignments == j]
                if len(pts_in_cluster) > 0:
                    wcss += float(np.sum((pts_in_cluster - centroids[j]) ** 2))

            # Centroids in original display scale
            if normalize:
                display_centroids = (centroids * ranges + min_vals).round(2).tolist()
            else:
                display_centroids = centroids.round(2).tolist()

            # D. Save iteration state
            iter_data = {
                "iteration": t,
                "centroids": display_centroids,
                "distance_matrix_D": dist_matrix.tolist(),
                "partition_matrix_U": partition_matrix.tolist(),
                "cluster_assignments": cluster_assignments.tolist(),
                "point_labels": point_labels,
                "points_x": X_orig[:, 0].round(2).tolist(),
                "points_y": X_orig[:, 1].round(2).tolist(),
                "metadata": metadata_list,
                "wcss": round(wcss, 4)
            }
            iterations.append(iter_data)

            # E. Compute New Centroids
            new_centroids = np.zeros((k, 2))
            for j in range(k):
                cluster_points = X_calc[cluster_assignments == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = cluster_points.mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            # Check convergence
            if np.allclose(centroids, new_centroids, atol=1e-4):
                converged = True

            centroids = new_centroids

        # 4. Generate Final Cluster Profiles / Summary Insights
        final_assignments = iterations[-1]["cluster_assignments"]
        final_centroids = iterations[-1]["centroids"]
        cluster_profiles = []

        for j in range(k):
            indices = [i for i, c in enumerate(final_assignments) if c == j]
            count = len(indices)
            pct = round((count / n_samples) * 100, 1) if n_samples > 0 else 0.0

            x_vals = [X_orig[i, 0] for i in indices] if count > 0 else [0.0]
            y_vals = [X_orig[i, 1] for i in indices] if count > 0 else [0.0]

            # Calculate Attrition Rate if available in metadata
            attrition_count = sum(
                1 for i in indices
                if str(metadata_list[i].get('attrition', '')).strip().lower() in ('yes', '1', 'true')
            )
            attrition_rate = round((attrition_count / count) * 100, 1) if count > 0 else 0.0

            cluster_profiles.append({
                "cluster_id": j + 1,
                "name": f"Cụm {j + 1}",
                "count": count,
                "percentage": pct,
                "centroid_x": final_centroids[j][0],
                "centroid_y": final_centroids[j][1],
                "min_x": round(float(np.min(x_vals)), 2) if count > 0 else 0.0,
                "max_x": round(float(np.max(x_vals)), 2) if count > 0 else 0.0,
                "mean_x": round(float(np.mean(x_vals)), 2) if count > 0 else 0.0,
                "min_y": round(float(np.min(y_vals)), 2) if count > 0 else 0.0,
                "max_y": round(float(np.max(y_vals)), 2) if count > 0 else 0.0,
                "mean_y": round(float(np.mean(y_vals)), 2) if count > 0 else 0.0,
                "attrition_count": attrition_count,
                "attrition_rate": attrition_rate
            })

        return {
            "k": k,
            "total_iterations": len(iterations),
            "converged": converged,
            "init_method": init_method,
            "normalized": normalize,
            "final_wcss": iterations[-1]["wcss"],
            "point_labels": point_labels,
            "iterations": iterations,
            "final_centroids": final_centroids,
            "cluster_profiles": cluster_profiles
        }

