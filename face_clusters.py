import face_recognition
import numpy as np
from face_loader import known_face_encodings

# cluster of nodes
def face_similarity_clusters(face_encodings, threshold=0.4):
    n = len(face_encodings)
    visited = [False] * n
    clusters = []
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(face_encodings[i] - face_encodings[j]) < threshold: # euclidean distance of 2 encoding
                adjacency[i].append(j)
                adjacency[j].append(i)
    def dfs(node, cluster): #dfs to check if face checked against all pics
        visited[node] = True
        cluster.append(node)
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                dfs(neighbor, cluster)
    for i in range(n):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)
    return clusters  # 

# quick search 
def binary_search_face(encoding):
    if not known_face_encodings:
        return -1
    distances = face_recognition.face_distance(known_face_encodings, encoding)
    min_distance = np.min(distances)
    if min_distance < 0.45:
        return np.argmin(distances) #min distances
    return -1