import face_recognition
import numpy as np
from face_loader import known_face_encodings , known_face_names

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