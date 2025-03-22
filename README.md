# Live Attendance System

A real-time face recognition-based attendance tracking system built with Streamlit and OpenCV.

## Overview

This application uses computer vision and face recognition to automatically detect students and mark their attendance in real-time. The system captures video from a webcam, identifies students based on their facial features, and maintains an attendance record that can be downloaded as an Excel file or emailed directly from the application.

## Features

- **Real-time Face Recognition**: Automatically identifies students from webcam feed
- **Attendance Tracking**: Marks students as present or absent with timestamps
- **10-Second Rule**: If a student is absent for more than 10 seconds, they are permanently marked absent
- **Excel Export**: Download attendance records as Excel files
- **Email Integration**: Send attendance reports directly via email
- **Face Encoding Caching**: Saves face encodings to improve startup performance

## Project Structure

```
├── app.py                 # Main application file
├── Students/              # Directory containing student face images
│   ├── Ashwin/            # Student 1 images
│   ├── Dhruv/             # Student 2 images
│   └── Prabhat/           # Student 3 images
├── face_encodings.pkl     # Cached face encodings
├── attendance.xlsx        # Generated attendance report
└── README.md              # This file
```

## Installation

1. Clone this repository or download the files

2. Install the required dependencies:

```bash
pip install streamlit opencv-python face-recognition numpy pandas yagmail
```

## Setup

1. **Student Images**: 
   - Create a folder named `Students` in the project directory
   - Inside this folder, create a subfolder for each student with their name
   - Add multiple face images of each student in their respective folders

2. **Register Numbers**:
   - Update the `register_numbers` dictionary in `app.py` with student names and their register numbers

3. **Email Configuration** (Optional):
   - Update the email credentials in the code if you want to use the email functionality
   - Replace `sender_email`, `sender_password`, and `receiver_email` with your own details

## Usage

1. Run the application:

```bash
streamlit run app.py
```

2. The application interface has three main buttons:
   - **Start**: Begin the webcam capture and face recognition
   - **Stop**: End the capture and display download/email options
   - **Reset**: Clear all attendance data and start fresh

3. Additional controls are available in the sidebar:
   - **Help**: View information about how the system works
   - **Regenerate Face Encodings**: Force regeneration of face encodings

## How It Works

1. **Face Detection and Recognition**:
   - The system loads face encodings from the `Students` folder
   - When a student is detected in the webcam feed, their face is compared to known encodings
   - If a match is found, the student is identified and marked present

2. **Attendance Rules**:
   - A student must be detected in 3 consecutive frames to be marked present
   - If a student disappears from view, they are immediately marked absent
   - If they return within 10 seconds, their present status is restored
   - If they remain absent for more than 10 seconds, they are permanently marked absent

3. **Attendance Export**:
   - After stopping the capture, the attendance report can be downloaded as an Excel file
   - The report includes student register numbers, names, attendance status, and timestamps
   - The report can also be emailed directly from the application

## Known Issues

- There is a typo in the code where `st.session_state.attendavnce` is used instead of `st.session_state.attendance` in some places

## Credits

Developed by Ashwin Suresh