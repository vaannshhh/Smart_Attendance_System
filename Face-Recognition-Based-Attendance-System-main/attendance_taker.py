import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import datetime
import requests

# Dlib / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        # Save the features of faces in the database
        self.face_features_known_list = []
        # Save the name of faces in the database
        self.face_name_known_list = []

        # List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        # Web App URL (Google Apps Script URL for the Spreadsheet)
        self.web_app_url = "https://script.google.com/macros/s/AKfycbzL8Td3VFGCJUWjb680MFbjCRI4l_NTGlhesGkSC12vHElMz2uBDH25MAoscBw2lXPY/exec"

        # Set to track names already marked for attendance
        self.marked_attendance_set = set()

    # Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database: %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])
                e_distance_current_frame_person_x_list.append(self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            name = self.current_frame_face_name_list[i]
            img_rd = cv2.putText(img_rd, name, tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')

        if name in self.marked_attendance_set:
            print(f"{name} is already marked as present for {current_date}")
            return  # Skip marking if already done
        
        params = {'name': name, 'date': current_date}

        try:
            response = requests.get(self.web_app_url, params=params)
            if response.status_code == 200:
                if 'already marked' in response.text:
                    print(f"{name} is already marked as present for {current_date}")
                else:
                    print(f"{name} marked as present for {current_date}")
                    self.marked_attendance_set.add(name)  # Add name to the set after marking
            else:
                print(f"Failed to mark attendance for {name}. Status Code: {response.status_code}")
        except Exception as e:
            print(f"Error occurred while marking attendance: {str(e)}")

    def process(self, stream):
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                faces = detector(img_rd, 0)
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                self.current_frame_face_centroid_list = []
                self.current_frame_face_name_list = []
                self.current_frame_face_feature_list = []

                if len(faces) != 0:
                    for i in range(len(faces)):
                        shape = predictor(img_rd, faces[i])
                        self.current_frame_face_centroid_list.append(
                            [int((faces[i].left() + faces[i].right()) / 2),
                             int((faces[i].top() + faces[i].bottom()) / 2)])

                        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
                        self.current_frame_face_feature_list.append(face_descriptor)

                    for k in range(len(faces)):
                        logging.debug("For face %d in camera: ", k + 1)
                        self.current_frame_face_X_e_distance_list = []

                        for i in range(len(self.face_features_known_list)):
                            distance = self.return_euclidean_distance(self.current_frame_face_feature_list[k],
                                                                      self.face_features_known_list[i])
                            self.current_frame_face_X_e_distance_list.append(distance)

                        similar_person_num = self.current_frame_face_X_e_distance_list.index(
                            min(self.current_frame_face_X_e_distance_list))
                        logging.debug("Minimum e-distance with %s: %f", self.face_name_known_list[similar_person_num],
                                      min(self.current_frame_face_X_e_distance_list))

                        if min(self.current_frame_face_X_e_distance_list) < 0.4:
                            self.current_frame_face_name_list.append(self.face_name_known_list[similar_person_num])
                            self.attendance(self.face_name_known_list[similar_person_num])
                        else:
                            self.current_frame_face_name_list.append("unknown")

                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                self.draw_note(img_rd)

                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.imshow("camera", img_rd)

        stream.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load camera or video stream for detection
    cap = cv2.VideoCapture(0)
    Face_Recognizer().process(cap)
