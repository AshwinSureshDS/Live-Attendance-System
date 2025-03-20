import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
import yagmail
import time
import pickle

# Cache face encodings for performance
@st.cache_data
def load_face_encodings(students_folder):
    encodings_file = "face_encodings.pkl"
    
    # Check if encodings file exists
    if os.path.exists(encodings_file):
        try:
            with open(encodings_file, "rb") as f:
                data = pickle.load(f)
                # Use auto-dismissing success message
                success_message = st.success("Successfully loaded face encodings from file!")
                time.sleep(1)  # Show message for just 1 second
                success_message.empty()  # Clear the message
                return data["encodings"], data["names"]
        except Exception as e:
            st.warning(f"Error loading encodings from file: {e}. Regenerating...")
    
    # If file doesn't exist or couldn't be loaded, generate encodings
    st.info("Generating face encodings... This may take a moment.")
    known_face_encodings = []
    known_face_names = []
    for person in os.listdir(students_folder):
        person_folder = os.path.join(students_folder, person)
        if os.path.isdir(person_folder):
            first_encoding = None
            for image_file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_file)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0 and first_encoding is None:
                    first_encoding = encodings[0]
            if first_encoding is not None:
                known_face_encodings.append(first_encoding)
                known_face_names.append(person)
    
    # Save encodings to file for future use
    try:
        with open(encodings_file, "wb") as f:
            pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        # Use auto-dismissing success message
        save_success = st.success("Face encodings saved to file for faster loading next time!")
        time.sleep(1)  # Show message for just 1 second
        save_success.empty()  # Clear the message
    except Exception as e:
        st.warning(f"Could not save encodings to file: {e}")
    
    return known_face_encodings, known_face_names

# Function to regenerate encodings (can be triggered by a button)
def regenerate_encodings(students_folder):
    encodings_file = "face_encodings.pkl"
    if os.path.exists(encodings_file):
        os.remove(encodings_file)
    return load_face_encodings(students_folder)

# STEP -1 : Loading known face encodings and names from the "students" folder
students_folder = "Students"
known_face_encodings, known_face_names = load_face_encodings(students_folder)

# STEP 2 - Define register numbers for each student
register_numbers = {
    "Ashwin": "2262207",
    "Dhruv": "2262217",
    "Prabhat": "2262266"
}

# STEP 3 - Sort student names by register numbers in ascending order
sorted_persons = sorted(register_numbers.keys(), key=lambda x: register_numbers[x])

# STEP 4 - Initialize attendance DataFrame and store in session state with sorted order and renamed column
if 'attendance' not in st.session_state:
    # Get current date and time
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%d/%m/%y")
    
    st.session_state.attendance = pd.DataFrame({
        "S.No.": list(range(1, len(sorted_persons) + 1)),
        "Reg. No.": [register_numbers[name] for name in sorted_persons],
        "Name": sorted_persons,
        "Attendance": ["Absent"] * len(sorted_persons),
        "Time": [None] * len(sorted_persons),
        "Date": [current_date] * len(sorted_persons)
    })
    
    # Remove permission state initialization

# STEP 5 - Set up counters for consecutive detections and a set for marked present students
if 'counters' not in st.session_state:
    st.session_state.counters = {name: 0 for name in sorted_persons}
if 'marked_present' not in st.session_state:
    st.session_state.marked_present = set()
# Track last seen time for each student
if 'last_seen' not in st.session_state:
    st.session_state.last_seen = {name: None for name in sorted_persons}
# Track original detection times
if 'detection_times' not in st.session_state:
    st.session_state.detection_times = {name: None for name in sorted_persons}
# Track permanent absence
if 'permanent_absent' not in st.session_state:
    st.session_state.permanent_absent = set()

# Initialize capture active state if not present
if 'capture_active' not in st.session_state:
    st.session_state.capture_active = False

# STEP 6 - Streamlit layout
st.title("Live Attendance System")

# Add custom CSS for better alignment and fixed credit in bottom right
st.markdown("""
<style>
    /* Any other custom styles can remain here */
    
    /* Fixed credit in bottom right corner */
    .footer {
        position: fixed;
        right: 10px;
        bottom: 10px;
        color: rgba(180, 180, 180, 0.7);
        font-size: 14px;
        z-index: 999;
        padding: 5px 10px;
        border-radius: 5px;
        background-color: rgba(0, 0, 0, 0.1);
    }
</style>

<div class="footer">Made by Ashwin Suresh</div>
""", unsafe_allow_html=True)

# Add sidebar elements
st.sidebar.title("Controls")

# Help button in sidebar
if 'show_help' not in st.session_state:
    st.session_state.show_help = False

# Use a unique key for the help button to prevent interference with other state
if st.sidebar.button("Help ❓", key="help_button"):
    st.session_state.show_help = not st.session_state.show_help

# Display help information when show_help is True
if st.session_state.show_help:
    st.sidebar.markdown("### Attendance System Help")
    st.sidebar.markdown("""
    **How the Attendance System Works:**
    
    1. **Face Detection:** The system automatically detects student faces and marks them present.
    
    2. **Attendance Status:**
       - When a student is detected, they are marked "Present" with the detection time.
       - If a student disappears, they are immediately marked "Absent".
    
    3. **10-Second Rule:**
       - If a student is absent for more than 10 seconds, they are permanently marked "Absent".
       - If they return within 10 seconds, their "Present" status is restored.
    
    4. **Downloading Attendance:**
       - After stopping the capture, you can download the attendance report as Excel.
       - You can also email the report directly from the application.
    """)

# Add button to regenerate encodings if needed
if st.sidebar.button("Regenerate Face Encodings"):
    with st.spinner("Regenerating face encodings..."):
        known_face_encodings, known_face_names = regenerate_encodings(students_folder)
    # Use auto-dismissing success message
    regen_success = st.sidebar.success("Face encodings regenerated successfully!")
    time.sleep(1)  # Show message for just 1 second
    regen_success.empty()  # Clear the message

# Add a function to handle the reset operation
def reset_attendance_data():
    """Reset all attendance data and state variables to start fresh."""
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%d/%m/%y")
    
    # Reset attendance DataFrame
    st.session_state.attendance = pd.DataFrame({
        "S.No.": list(range(1, len(sorted_persons) + 1)),
        "Reg. No.": [register_numbers[name] for name in sorted_persons],
        "Name": sorted_persons,
        "Attendance": ["Absent"] * len(sorted_persons),
        "Time": [None] * len(sorted_persons),
        "Date": [current_date] * len(sorted_persons)
    })
    
    # Reset all tracking variables
    st.session_state.counters = {name: 0 for name in sorted_persons}
    st.session_state.marked_present = set()
    st.session_state.last_seen = {name: None for name in sorted_persons}
    st.session_state.detection_times = {name: None for name in sorted_persons}
    st.session_state.permanent_absent = set()
    
    # Reset capture state if active
    if st.session_state.capture_active:
        st.session_state.capture_active = False
    
    # Reset stop trigger
    st.session_state.stop_triggered = False
    if 'stop_button_clicked' in st.session_state:
        st.session_state.stop_button_clicked = False

# STEP 7 - Buttons for starting and stopping the capture
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Start"):
        st.session_state.capture_active = True
with col2:
    if st.button("Stop"):
        st.session_state.capture_active = False
        st.session_state.stop_button_clicked = True  # Set flag when Stop button is clicked
with col3:
    if st.button("Reset"):
        # Call the reset function
        reset_attendance_data()
        # Show success message
        st.success("All data has been reset. You can start a new session.")

# STEP 8 - Placeholder for video frames and attendance table
frame_placeholder = st.empty()  # Placeholder for video frames
attendance_placeholder = st.empty()  # Placeholder for attendance table

# STEP 9 - Initialize video capture only when capture is active
video_capture = None  # Initialize as None

if st.session_state.capture_active:
    # Start video capture
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        st.error("Failed to capture video. Check your camera.")
        st.session_state.capture_active = False
    else:
        while st.session_state.capture_active:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to capture video. Check your camera.")
                break

            # Resize frame to 25% for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            detected_names = []

            # Process each detected face
            for i, face_encoding in enumerate(face_encodings):
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(distances) > 0:
                    min_distance_index = np.argmin(distances)
                    # Threshold of 0.6 for face match
                    if distances[min_distance_index] < 0.6:
                        name = known_face_names[min_distance_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"
                detected_names.append(name)

                # Scale face locations back to original frame size
                top, right, bottom, left = [int(x * 4) for x in face_locations[i]]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Update attendance based on consecutive detections and absences
            unique_detected_names = set(detected_names)
            current_time = time.time()
            
            for name in st.session_state.counters:
                # Remove permission check
                
                if name in unique_detected_names:
                    # Student is currently visible
                    st.session_state.counters[name] += 1
                    st.session_state.last_seen[name] = current_time
                    
                    # Mark present after 3 consecutive frames (faster update)
                    if st.session_state.counters[name] >= 3 and name not in st.session_state.permanent_absent:
                        if name not in st.session_state.marked_present:
                            # First time detection
                            st.session_state.marked_present.add(name)
                            st.session_state.attendance.loc[st.session_state.attendance["Name"] == name, "Attendance"] = "Present"
                            detection_time = datetime.now()
                            st.session_state.detection_times[name] = detection_time
                            st.session_state.attendavnce.loc[st.session_state.attendance["Name"] == name, "Time"] = detection_time.strftime("%I:%M:%S %p")
                        elif name in st.session_state.marked_present and st.session_state.attendance.loc[st.session_state.attendance["Name"] == name, "Attendance"].values[0] == "Absent":
                            # Student returned within 10 seconds
                            st.session_state.attendance.loc[st.session_state.attendance["Name"] == name, "Attendance"] = "Present"
                            # Restore original detection time
                            if st.session_state.detection_times[name]:
                                st.session_state.attendance.loc[st.session_state.attendance["Name"] == name, "Time"] = st.session_state.detection_times[name].strftime("%I:%M:%S %p")
                            # If they returned, remove from permanent absent list
                            if name in st.session_state.permanent_absent:
                                st.session_state.permanent_absent.remove(name)
                else:
                    # Student is not visible
                    st.session_state.counters[name] = 0
                    
                    # If student was present but now absent
                    if name in st.session_state.marked_present and st.session_state.attendance.loc[st.session_state.attendance["Name"] == name, "Attendance"].values[0] == "Present":
                        # Mark as absent immediately
                        st.session_state.attendance.loc[st.session_state.attendance["Name"] == name, "Attendance"] = "Absent"
                        st.session_state.attendance.loc[st.session_state.attendance["Name"] == name, "Time"] = None
                    
                    # Check if student has been absent for more than 10 seconds
                    if name in st.session_state.marked_present and st.session_state.last_seen[name] is not None:
                        time_since_last_seen = current_time - st.session_state.last_seen[name]
                        if time_since_last_seen > 10:  # More than 10 seconds
                            # Mark as permanently absent
                            st.session_state.permanent_absent.add(name)
                            st.session_state.attendance.loc[st.session_state.attendance["Name"] == name, "Attendance"] = "Absent"
                            st.session_state.attendance.loc[st.session_state.attendance["Name"] == name, "Time"] = None

            # Display the original frame with annotations
            frame_placeholder.image(frame, channels="BGR")
            
            # Display attendance dataframe without permission column
            attendance_placeholder.dataframe(
                st.session_state.attendance,
                column_config={
                    "S.No.": st.column_config.NumberColumn(width="small"),
                    "Reg. No.": st.column_config.TextColumn(width="small"),
                    "Name": st.column_config.TextColumn(width="small"),
                    "Attendance": st.column_config.TextColumn(width="small"),
                    "Time": st.column_config.TextColumn(width="small"),
                    "Date": st.column_config.TextColumn(width="small")
                },
                hide_index=True,
                use_container_width=True
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Optional: Allow quitting with 'q' key
                st.session_state.capture_active = False
                break

# Release the camera when capture is stopped or app exits
if video_capture is not None and not st.session_state.capture_active:
    video_capture.release()

# STEP 10 - Show download and email options only after stopping capture (triggered by Stop button)
# Make sure this is only triggered by the Stop button, not by the help button
if not st.session_state.capture_active and st.session_state.get('stop_triggered', False):
    if 'attendance' in st.session_state:
        final_attendance = st.session_state.attendance
        
        # Display the final attendance with proper column widths
        st.dataframe(
            final_attendance,
            column_config={
                "S.No.": st.column_config.NumberColumn(width="small"),
                "Reg. No.": st.column_config.TextColumn(width="small"),
                "Name": st.column_config.TextColumn(width="small"),
                "Attendance": st.column_config.TextColumn(width="small"),
                "Time": st.column_config.TextColumn(width="small"),
                "Date": st.column_config.TextColumn(width="small")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Save attendance to Excel - no permission column
        final_attendance.to_excel("attendance.xlsx", index=False)

        # Download button for the Excel file
        with open("attendance.xlsx", "rb") as file:
            st.download_button(
                label="Download Attendance as Excel",
                data=file,
                file_name="attendance.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # Email functionality
        sender_email = "ashwinsuresh2004.mp@gmail.com"  # Replace with your email
        sender_password = "jbqp umbg cqqk epgm"      # Replace with your email password or App Password
        receiver_email = "ashwin.suresh@btech.christuniversity.in"  # Replace with recipient email

        if st.button("Send Attendance via Email"):
            try:
                yag = yagmail.SMTP(sender_email, sender_password)
                yag.send(
                    to=receiver_email,
                    subject="Attendance Report",
                    contents="Please find the attendance report attached.",
                    attachments="attendance.xlsx"
                )
                st.success("Email sent successfully!")
            except Exception as e:
                st.error(f"Failed to send email: {e}")

    # Reset stop_triggered after showing buttons (optional, to allow re-triggering)
    st.session_state.stop_triggered = False

# Set stop_triggered when Stop button is clicked - make sure this logic is separate from the help button
if st.session_state.capture_active and 'stop_triggered' not in st.session_state:
    st.session_state.stop_triggered = False
elif not st.session_state.capture_active and st.session_state.get('stop_triggered', False) is False:
    # This should only be triggered by the Stop button
    if 'stop_button_clicked' in st.session_state and st.session_state.stop_button_clicked:
        st.session_state.stop_triggered = True
        st.session_state.stop_button_clicked = False

print("Streamlit app file 'app.py' has been created. To run it, open a terminal in this directory and type:")
print("streamlit run app.py")
