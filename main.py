import cv2
import face_recognition
from face_loader import load_known_faces, sort_known_faces, known_face_names
from face_clusters import face_similarity_clusters, binary_search_face
from face_compare import dp_face_compare, greedy_face_assignment


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
            cv2.putText(frame, name, (left, top - 8), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release() # stops webcam recording
    cv2.destroyAllWindows()  # closes webcam window

main()

