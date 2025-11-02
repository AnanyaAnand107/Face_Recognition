import cv2
import face_recognition
import numpy as np
import os

# data structure for known face
known_face_encodings = []
known_face_names = []

# loading known face
def load_known_faces(directory):
    # Scan for all images in the directory (.jpg, .jpeg, .png)
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(directory, filename)
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_face_encodings.append(encodings[0])
                # Use filename as name
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                print(f"Loaded face: {name}")
            else:
                print(f"Could not encode: {filename}")
    print("All loaded faces:", known_face_names)

def sort_known_faces():
    global known_face_encodings, known_face_names
    zipped = list(zip(known_face_encodings, known_face_names))
    zipped.sort(key=lambda x: x[1].lower())  # sort by name aplhabettically
    if zipped:
        enc_sorted, names_sorted = zip(*zipped)
        known_face_encodings = list(enc_sorted)
        known_face_names = list(names_sorted)

# face encoding comparison 
compare = {}
def dp_face_compare(encoding):
    key = encoding.tobytes()
    if key in compare:
        return compare[key]
    results = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.45)
    compare[key] = results
    return results

# nearest match 
def greedy_face_assignment(face_encoding):
    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    print("Distances:", distances)  # Debug!
    min_distance = min(distances) if len(distances) > 0 else None
    if min_distance is not None and min_distance < 0.045:  # threshold 
        index = np.argmin(distances)
        return known_face_names[index]
    return "Unknown"

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


def main():
    load_known_faces("known_faces")
    sort_known_faces()
    print(f"Known faces loaded: {known_face_names}")

    video_capture = cv2.VideoCapture(0) # video captures live image 

    while True:
        web_cam, frame = video_capture.read()
        if not web_cam:
            print("Failed to grab frame from webcam")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #BGR to RGB

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        print("Detected faces:", len(face_encodings)) #face count 
        if len(face_encodings) > 1:
            clusters = face_similarity_clusters(face_encodings) 
            print("Face clusters detected:", clusters)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            match_results = dp_face_compare(encoding)
            print("Match results for this face:", match_results)
            name = greedy_face_assignment(encoding)
            search_index = binary_search_face(encoding)
            if search_index != -1:
                name = known_face_names[search_index]

            print("Detected name:", name) # recognisation 
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (right, top - 8), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release() # stops webcam recording
    cv2.destroyAllWindows()  # closes webcam window

main()
