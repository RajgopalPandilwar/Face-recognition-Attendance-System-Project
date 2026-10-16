"""
Face Recognition Attendance System - Main Module
==================================================
Author: Rajgopal Pandilwar

This module implements a real-time face recognition attendance system.
It captures video from the webcam, detects and recognizes faces, and marks
attendance in a CSV file for recognized students.

Usage:
    python main.py

Press Enter to exit the application.
"""

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# ============================================================
# Configuration
# ============================================================
IMAGE_PATH = 'Images_Attendance'  # Directory containing student images
ATTENDANCE_FILE = 'Attendance.csv'  # CSV file to store attendance records


def load_known_faces(image_dir):
    """
    Load and encode all student images from the given directory.

    Args:
        image_dir (str): Path to the directory containing student images.

    Returns:
        tuple: (list of encoded faces, list of student names)
    """
    images = []
    class_names = []

    # List all image files in the directory
    image_files = os.listdir(image_dir)
    print(f"Found {len(image_files)} images: {image_files}")

    for file_name in image_files:
        # Read each image
        img = cv2.imread(os.path.join(image_dir, file_name))
        images.append(img)
        # Extract student name from filename (without extension)
        class_names.append(os.path.splitext(file_name)[0])

    return images, class_names


def find_face_encodings(images):
    """
    Generate face encodings for a list of images.

    Args:
        images (list): List of BGR images (numpy arrays).

    Returns:
        list: List of 128-dimensional face encodings.
    """
    encode_list = []
    for img in images:
        # Convert BGR to RGB (face_recognition expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Get the first face encoding found in the image
        encode = face_recognition.face_encodings(img_rgb)[0]
        encode_list.append(encode)
    return encode_list


def mark_attendance(name):
    """
    Mark a student's attendance in the CSV file.
    Prevents duplicate entries for the same student.

    Args:
        name (str): Name of the student to mark present.
    """
    with open(ATTENDANCE_FILE, 'r+') as f:
        # Read existing attendance records
        data_lines = f.readlines()
        name_list = []
        for line in data_lines:
            entry = line.split(',')
            name_list.append(entry[0])
        # Only mark if not already present
        if name not in name_list:
            now = datetime.now()
            time_string = now.strftime('%H:%M:%S')
            date_string = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{time_string},{date_string}')
            print(f"Attendance marked for: {name}")


def main():
    """Main function to run the face recognition attendance system."""
    # Step 1: Load known faces
    images, class_names = load_known_faces(IMAGE_PATH)

    # Step 2: Encode known faces
    known_encodings = find_face_encodings(images)
    print(f"Encoding complete. {len(known_encodings)} faces registered.")

    # Step 3: Start webcam capture
    cap = cv2.VideoCapture(0)
    print("Webcam started. Press Enter to exit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame from webcam.")
            break

        # Resize for faster processing (1/4 resolution)
        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        # Detect faces and compute encodings for current frame
        face_locations = face_recognition.face_locations(img_small)
        face_encodings = face_recognition.face_encodings(img_small, face_locations)

        # Compare each detected face with known encodings
        for encode_face, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encode_face)
            face_distances = face_recognition.face_distance(known_encodings, encode_face)

            # Find the best match (minimum distance)
            match_index = np.argmin(face_distances)

            if matches[match_index]:
                name = class_names[match_index].upper()
                print(f"Recognized: {name}")

                # Scale coordinates back to original frame size
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Draw rectangle and name label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 250, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Mark attendance for recognized student
                mark_attendance(name)

        # Display the output frame
        cv2.imshow('Face Recognition Attendance', img)

        # Exit on Enter key
        if cv2.waitKey(10) == 13:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Attendance system stopped.")


if __name__ == '__main__':
    main()
