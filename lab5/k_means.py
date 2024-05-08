import numpy as np


def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids


def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    number_of_centroids = 1
    while (number_of_centroids != k):
        max_distance = 0.0
        index = -1
        for i in range(0, data.shape[0]):
            distance = np.sqrt(np.sum((data[i] - centroids[: number_of_centroids]) ** 2))
            if (distance > max_distance):
                index = i
                max_distance = distance
        centroids[number_of_centroids] = data[index]
        number_of_centroids += 1

    print(centroids)
    return centroids


def initialize_centroids_kmeans_pp_correct(data, k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    number_of_centroids = 1
    while number_of_centroids != k:
        max_distance = 0.0
        index = -1
        for i in range(0, data.shape[0]):
            # Find: min dist between element and centroid
            distances = [np.sqrt(np.sum((data[i] - centroids[j]) ** 2)) for j in
                         range(number_of_centroids)]
            min_distance = min(distances)
            if min_distance > max_distance:
                index = i
                max_distance = min_distance
        centroids[number_of_centroids] = data[index]
        number_of_centroids += 1
    return centroids


def assign_to_cluster(data, centroids):
    assignments = np.zeros(data.shape[0], dtype=int)
    for i in range(0, data.shape[0]):
        distance_max = float("inf")
        index = 0
        for j in range(0, centroids.shape[0]):
            distance = np.sqrt(np.sum((data[i] - centroids[j, :]) ** 2))
            if distance_max > distance:
                distance_max = distance
                index = j
        assignments[i] = int(index)
    return assignments


def update_centroids(data, assignments, k):
    # TODO find new centroids based on the assignments
    centroids = np.zeros((k, data.shape[1]))
    for i in range(0, k):
        assigned_data = data[assignments == i]
        centroids[i] = np.mean(assigned_data, axis=0)
    return centroids


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))


def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp_correct(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # max number of iteration = 100
        # print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments, num_centroids)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)
