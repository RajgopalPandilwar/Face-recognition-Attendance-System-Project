"""
Face Recognition Attendance System - Demo Module
==================================================
Author: Rajgopal Pandilwar
Based on original work by Vatshayan

This is a simplified demo module that demonstrates face recognition
between two static images (without webcam). Useful for testing the
face_recognition library setup.

Usage:
    python AttendanceProject.py

Note: Ensure the image files exist in Images_Attendance/ directory.
"""

import cv2
import numpy as np
import face_recognition


def compare_two_faces():
    """
    Compare two face images and display recognition results.
    This is a basic demo to verify face_recognition is working.
    """
    # Load the known face image
    img_known = face_recognition.load_image_file('Images_Attendance/modi-image-for-InUth.jpg')
    img_known = cv2.cvtColor(img_known, cv2.COLOR_BGR2RGB)

    # Load the test face image
    img_test = face_recognition.load_image_file('Images_Attendance/narendra-modi.jpg')
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

    # Detect face location and compute encoding for known image
    face_loc_known = face_recognition.face_locations(img_known)[0]
    encode_known = face_recognition.face_encodings(img_known)[0]

    # Draw a rectangle around the known face
    cv2.rectangle(img_known,
                  (face_loc_known[3], face_loc_known[0]),
                  (face_loc_known[1], face_loc_known[2]),
                  (155, 0, 255), 2)

    # Detect face location and compute encoding for test image
    face_loc_test = face_recognition.face_locations(img_test)[0]
    encode_test = face_recognition.face_encodings(img_test)[0]

    # Draw a rectangle around the test face
    cv2.rectangle(img_test,
                  (face_loc_test[3], face_loc_test[0]),
                  (face_loc_test[1], face_loc_test[2]),
                  (155, 0, 255), 2)

    # Compare the two faces
    results = face_recognition.compare_faces([encode_known], encode_test)
    face_distance = face_recognition.face_distance([encode_known], encode_test)

    # Display match result on the test image
    cv2.putText(img_test,
                f'Match: {results[0]} Distance: {round(face_distance[0], 2)}',
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    # Show both images
    cv2.imshow('Known Face', img_known)
    cv2.imshow('Test Face', img_test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Match Result: {results[0]}")
    print(f"Face Distance: {round(face_distance[0], 2)}")


if __name__ == '__main__':
    compare_two_faces()
