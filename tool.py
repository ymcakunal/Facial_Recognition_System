#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import cv2
import os
import shutil
import face_recognition
import numpy as np
import zipfile
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image as PilImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QMessageBox, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt
import subprocess

class VideoLimitApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 600, 400)
        self.setWindowTitle("Face Recognition Tool")
        self.setFixedSize(600, 400)

        icon = QIcon("National_Forensic_Sciences_University_Logo.png")
        self.setWindowIcon(icon)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        image_label = QLabel(self)
        image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap("National_Forensic_Sciences_University_Logo.png")
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)

        text_label = QLabel(self)
        text_label.setAlignment(Qt.AlignLeft)
        text_label.setFont(QFont("Arial", 12, QFont.Bold))
        text_label.setText("Face Recognition System\nSelect a video and click 'Submit' to continue.")
        text_label.setWordWrap(True)

        select_video_button = QPushButton("Select Video", self)
        select_video_button.clicked.connect(self.select_video)

        self.submit_button = QPushButton("Submit", self)
        self.submit_button.clicked.connect(self.submit_video)
        self.submit_button.setDisabled(True)

        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.cancel_input)

        layout.addWidget(image_label)
        layout.addWidget(text_label)
        layout.addWidget(select_video_button)
        layout.addWidget(self.submit_button)
        layout.addWidget(cancel_button)

        central_widget.setLayout(layout)

    def select_video(self):
        options = QFileDialog.Options()
        video_path, _ = QFileDialog.getOpenFileName(self, "Select a Video", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)

        if video_path:
            self.selected_video_path = video_path
            self.submit_button.setEnabled(True)

    def submit_video(self):
        if hasattr(self, "selected_video_path"):
            video_path = self.selected_video_path
            QMessageBox.information(self, "Submission", "Video submitted successfully. Processing will start shortly.")

            # Begin the backend processing
            self.run_backend(video_path)

        else:
            QMessageBox.warning(self, "No Video Selected", "Please select a video before submitting.")

    def cancel_input(self):
        sys.exit()

    def run_backend(self, video_path):
        # Define the video file path
        video_path = video_path

        # Create directories to save the captured images and matched faces
        output_directory = 'captured_images'
        faces_directory = 'detected_faces'
        matched_faces_directory = 'matched_faces'

        # Define the frame skip interval (e.g., 5 captures every 5th frame)
        frame_skip = 5

        # Function to empty the captured_images, detected_faces, and matched_faces folders
        def clear_output_directory(directory):
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

        clear_output_directory(output_directory)
        clear_output_directory(faces_directory)
        clear_output_directory(matched_faces_directory)

        # Unzip the folder containing known faces
        known_faces_zip = 'KF2.zip'
        known_faces_dir = 'Face Detect/'

        with zipfile.ZipFile(known_faces_zip, 'r') as zip_ref:
            zip_ref.extractall(known_faces_dir)

        # Unzip the text folder
        text_folder_zip = 'txt.zip'
        text_folder = 'TextFiles/'

        with zipfile.ZipFile(text_folder_zip, 'r') as zip_ref:
            zip_ref.extractall(text_folder)

        # Load a pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_count = 0
        captured_frame_count = 0
        matched_image_saved = False

        known_faces = []

        # Load known faces (provide paths to known face images)
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
                face_encoding = face_recognition.face_encodings(image)[0]
                known_faces.append((face_encoding, filename))

        # Create a PDF to store matched faces and text content
        pdf_filename = 'matched_faces_info.pdf'
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        story = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_skip == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for i, (x, y, w, h) in enumerate(faces):
                    face = frame[y:y + h, x:x + w]
                    face_filename = os.path.join(faces_directory, f"face_{captured_frame_count:04d}_{i:02d}.jpg")
                    cv2.imwrite(face_filename, face)

                    unknown_image = face_recognition.load_image_file(face_filename)
                    unknown_encoding = face_recognition.face_encodings(unknown_image)

                    if unknown_encoding:
                        match_results = face_recognition.compare_faces([face[0] for face in known_faces], unknown_encoding[0])

                        if True in match_results and not matched_image_saved:
                            print("Match found for face in frame:", captured_frame_count)
                            matched_face_index = match_results.index(True)
                            matched_face = known_faces[matched_face_index]
                            matched_face_filename = os.path.join(matched_faces_directory, matched_face[1])
                            shutil.copy(os.path.join(known_faces_dir, matched_face[1]), matched_face_filename)
                            matched_image_saved = True

                            # Add matched image and corresponding text content to the PDF
                            text_filename = os.path.splitext(matched_face[1])[0] + '.txt'
                            text_file_path = os.path.join(text_folder, text_filename)

                            if os.path.isfile(text_file_path):
                                # Load image
                                img = Image(matched_face_filename, width=400, height=400)
                                story.append(img)

                                # Load and add text content
                                with open(text_file_path, 'r') as text_file:
                                    text_content = text_file.read()
                                text = Paragraph(text_content, getSampleStyleSheet()['Normal'])
                                story.append(Spacer(1, 12))
                                story.append(text)

                captured_frame_count += 1

            frame_count += 1

        cap.release()

        # Build the PDF
        doc.build(story)

        print(f"{captured_frame_count} frames captured and saved in '{output_directory}'")
        print(f"{captured_frame_count * len(faces)} faces detected and saved in '{faces_directory}'")

        # Check if matched_faces_info.pdf is generated
        if os.path.exists(pdf_filename):
            reply = QMessageBox.question(None, "Processing Completed",
                                         "Matches found and results saved in 'matched_faces_info.pdf'.\n"
                                         "Do you want to save the PDF?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

            if reply == QMessageBox.Yes:
                options = QFileDialog.Options()
                save_path, _ = QFileDialog.getSaveFileName(None, "Save PDF", "matched_faces_info.pdf", "PDF Files (*.pdf);;All Files (*)", options=options)
                if save_path:
                    shutil.move(pdf_filename, save_path)
            elif reply == QMessageBox.No:
                pass  # User clicked "No," do nothing
            else:
                pass  # User clicked "Cancel," do nothing
        else:
            QMessageBox.information(None, "Processing Completed", "No matches found.")

def main():
    app = QApplication(sys.argv)
    window = VideoLimitApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

