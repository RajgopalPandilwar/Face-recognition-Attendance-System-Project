# Face Recognition Attendance System

**Author:** Rajgopal Pandilwar

Final Year Face Recognition Attendance System Project.

**YouTube Video:** [Implementation Video](https://youtu.be/tLhFaAurhGw)

---

## Abstract

The management of attendance can be a great burden on teachers if done by hand. To resolve this problem, a smart and automated attendance management system is being utilized. By utilizing this framework, the problem of proxies and students being marked present even though they are not physically present can easily be solved.

This system marks attendance using a live video stream. The frames are extracted from video using OpenCV. The main implementation steps used in this type of system are:

1. **Face Detection** — Detect faces in the video stream
2. **Face Recognition** — Recognize the detected faces using the `face_recognition` library
3. **Attendance Marking** — Compare recognized faces with the database and mark attendance

After recognition, attendance is recorded in a CSV file. This model is a successful technique to manage the attendance of students.

---

## How It Works

1. Student images are stored in the `Images_Attendance` directory
2. The system encodes all known faces at startup
3. Webcam captures live video frames
4. Detected faces are compared against known encodings
5. Recognized students are marked present in `Attendance.csv`

---

## Requirements

```bash
pip install opencv-python
pip install face-recognition
pip install numpy
```

---

## How to Run

```bash
python main.py
```

Press **Enter** to exit the application.

---

## Reference Research Paper

[IEEE Xplore Document 9215441](https://ieeexplore.ieee.org/document/9215441)

---

## How to Reach the Author

- **GitHub:** [RajgopalPandilwar](https://github.com/RajgopalPandilwar)

---

## Credits

> YouTube explanation: https://youtu.be/tLhFaAurhGw

---

*This project has been customized and maintained by Rajgopal Pandilwar.*
