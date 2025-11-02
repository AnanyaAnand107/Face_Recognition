import os
import face_recognition

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