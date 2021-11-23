import dlib
from tqdm import tqdm
import os
import cv2
from imutils import face_utils
import numpy as np
import ipdb


## Step 2: parse landmark

def get_bbox_landmark(face_path, detector, predictor):
    img = cv2.imread(face_path)
    dets = detector(img, 1)
    if len(dets) == 0:
        bbox = [24, 32, 87, 94]
        landmarks = np.array([[77, 46],
                [66, 48],
                [36, 54],
                [47, 51],
                [62, 70]])
    for k, d in enumerate(dets):
        
        bbox = [d.left(), d.top(), d.right(), d.bottom()]
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        landmarks = face_utils.shape_to_np(shape)

    return bbox, landmarks

def save_bbox_lnmk(dataset_root):

    #Dlib facial landmarks model的path
    predictor_path = "landmark/models/shape_predictor_5_face_landmarks.dat"

    #detector為臉孔偵測，predictor為landmarks偵測
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for folder in tqdm(os.listdir(dataset_root)):
        if not os.path.isdir(os.path.join(dataset_root, folder)):
            continue
        fold = dataset_root + '/' + folder
        onlyfiles = []
        for f in os.listdir(fold):
            if os.path.splitext(f)[1] == '.jpg':
                img_file = fold + '/' + f
                bbox, landmarks = get_bbox_landmark(img_file, detector, predictor)
                save_file_bbox = img_file + '_bbox.npy'
                save_file_lnmk = img_file + '_lnmk.npy'
                #print(save_file_lnmk)
                np.save(save_file_bbox, np.array(bbox))
                np.save(save_file_lnmk, np.array(landmarks))
                #print(bbox, '\n',  landmarks)
                

if __name__ == '__main__':
    # dataset_root = '../data/test/closed_set/test_pairs_align'
    # dataset_root = '../data/train/val_split'
    dataset_root = '../data/test/open_set/unlabeled_data_align'
    save_bbox_lnmk(dataset_root)