import dsw_python
import cv2
import numpy as np
import os


class Label():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.l_type = ""
        self.occlusion = 0
        self.best_overlap = 0.

class Hyp():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        
class Calib():
    def __init__(self):
        self.p_left = np.zeros((3,4))        
        self.p_right = np.zeros((3,4))
        self.k_left = np.zeros((3,3))
        self.r0 = np.zeros((3,3))
        self.tx = 0.

def readKittiLabels(path):
    label_list = []
    with open(path) as f:
        
        for line in f:
            l = Label()
            lines = line.split()
            for idx, token in enumerate(lines):

            # read first two tokens separately
                if idx == 0:
                    l.l_type = str(token)
                elif idx == 2:
                    l.occlusion = int(token)
                elif idx == 4:
                    l.x = int(float(token))
                    l.y = int(float(lines[idx+1]))
                    l.w = int(float(lines[idx+2])) - int(float(token))
                    l.h = int(float(lines[idx+3])) - int(float(lines[idx+1]))
                    break
            label_list.append(l)
    return label_list

def readKittiCalib(path):
    
    calib = Calib()
    with open(path) as f:

        for line in f:
            l = Label()
            lines = line.split()
            vector = np.asarray(lines[1:],dtype=np.float32)
            if len(lines) == 0:
                continue
                   
            if lines[0] == "P2:":
                calib.p_left = vector.reshape(3,4)
            elif lines[0] == "P3:":  
                calib.p_right = vector.reshape(3,4)
            elif lines[0] == "R0_rect:":
                calib.r0 = vector.reshape(3,3)
    
    calib.tx = float(calib.p_right[0,3])
    rot = np.zeros((3,3))
    trans = np.zeros((4,1))
    cam_matrix = np.zeros((3,3),dtype= np.double)
    cam_matrix = cv2.decomposeProjectionMatrix(calib.p_left, cam_matrix, rot, trans)
    calib.k_left = cam_matrix[0]
    return calib

def union( a, b):

    x = min(a.x, b.x)
    y = min(a.y, b.y)
    w = max(a.x + a.w, b.x + b.w) - x
    h = max(a.y + a.h, b.y + b.h) - y
    return w * h

def intersection( a, b):

    x = max(a.x, b.x)
    y = max(a.y, b.y)
    w = min(a.x + a.w, b.x + b.w) - x
    h = min(a.y + a.h, b.y + b.h) - y
    if w < 0 or h < 0:
      return 0.0  
    return w * h

def bb_intersection_over_union( label, hyp):

    # intersection
    num = float(intersection(label, hyp))
    # union
    den = float(union(label, hyp))
    # intersection/union
    iou = num / den

    # return the intersection over union value
    return iou              

if __name__ == "__main__":

    # Paths for Kitti Object detection files
    kittiLeftImagePath = "/scratch/fs2/KITTI/data_object_image_2/training/image_2"
    kittiRightImagePath = "/scratch/fs2/KITTI/data_object_image_3/training/image_3"
    kittiCalibPath = "/scratch/fs2/KITTI/data_object_calib/training/calib"
    kittiLabelPath = "/scratch/fs2/KITTI/training/label_2"
    kittiDispPath = "/scratch/fs2/KITTI/disparities"

    # Empty lists for images and calib
    leftImgPathList = []
    rightImgPathList = []
    calibPathList = []
    labelPathList = []
    dispPathList = []

        # Get list of all left image paths
    for file in sorted(os.listdir(kittiLeftImagePath)):
        if file.endswith(".png"):
            leftImgPathList.append(os.path.join(kittiLeftImagePath,file))
            
    # Get list of all right image paths
    for file in sorted(os.listdir(kittiRightImagePath)):
        if file.endswith(".png"):
            rightImgPathList.append(os.path.join(kittiRightImagePath,file))

    # Get list of all calib paths
    for file in sorted(os.listdir(kittiCalibPath)):
        if file.endswith(".txt"):
            calibPathList.append(os.path.join(kittiCalibPath,file))
            
    # Get list of all label paths
    for file in sorted(os.listdir(kittiLabelPath)):
        if file.endswith(".txt"):
            labelPathList.append(os.path.join(kittiLabelPath,file)) 
            
    # Get list of all disparity images
    for file in sorted(os.listdir(kittiDispPath)):
        if file.endswith(".png"):
            dispPathList.append(os.path.join(kittiDispPath,file)) 


    # SGM parameters
    minDisparity = 2
    numDisparities = 114 - minDisparity
    windowSize = 5
    p1 = 8*3*windowSize**2
    p2 = 32*3*windowSize**2
    dispMaxDiff = 1
    preFilterCap = 0
    uniqueRatio = 10
    speckleWindowSize = 100
    speckeRange = 32
    fullDP = False


    # SGM Constructor
    sgm = cv2.StereoSGBM(minDisparity, numDisparities, windowSize, p1, p2, dispMaxDiff, preFilterCap, uniqueRatio, speckleWindowSize,   speckeRange, fullDP)

    # DSW parameters
    objectWidth = 0.6
    objectHeight = 1.73
    hypAspect = 2.88
    hypMinWidth = 10
    hypMaxWidth = 200
    hypClassId = 10
    maxNans = 6
    maxStddev = 0.05
    stepPerc = 0.3
    homogeneity_method = 1

    dsw = dsw_python.DisparitySlidingWindow(objectWidth, objectHeight, hypAspect, hypMinWidth, hypMaxWidth, hypClassId, maxNans, maxStddev, stepPerc, homogeneity_method)

    # for each frame 
    for label_path, left_img_path, right_img_path, calib_path, disp_path in zip(labelPathList, leftImgPathList, rightImgPathList, calibPathList, dispPathList):

        # Read KITTI labels
        label_list = readKittiLabels(label_path)
        ped_in_frame = False

        # Check if pedestrian is in frame
        for l in label_list:
            if l.l_type == "Pedestrian":
                ped_in_frame = True

        # We only evaulate Pedestrians
        if ped_in_frame:
            
            #read calib and images
            calib = readKittiCalib(calib_path)
            img_left = cv2.imread(left_img_path, -1)
            
            img_right = cv2.imread(right_img_path, -1)

            # calculate disparity by SGM
            disparity = sgm.compute(img_left, img_right)
            disparity = np.uint16(disparity)
            disparity = disparity.astype(np.float32) /16.0
           
            # disp=1 is invalid
            disparity[disparity==1.0]=np.nan

            # init lookup-table
            dist = np.zeros((5,1))
            rect_list = np.zeros((4,319))               
            dsw.initLookUpTable(calib.tx, calib.k_left, dist, 0, 114, 1./16. )
            
            # do DSW und return rect list of ints
            hyps_list = []
            rect_list = dsw.generate_py(disparity, calib.tx)

            # Go through all labels in file
            for l in label_list:
                # We only evaluate pedestrians
                if l.l_type == "Pedestrian":
                    best_ov = 0.
                    best_idx = -1
                    cv2.rectangle(img_left, (l.x, l.y), (l.x + l.w, l.y + l.h), (255,255,255), 2)

                    # find proposal with best overlap
                    for i in xrange(0, len(rect_list), 4):
                        h=Hyp()
                        h.x = rect_list[i]
                        h.y = rect_list[i+1]
                        h.w = rect_list[i+2] 
                        h.h = rect_list[i+3]
                        iou = bb_intersection_over_union(l,h)
                        if iou > best_ov:
                            best_ov = iou
                            best_idx = i
                    # plot best overlap
                    if best_idx is not -1:
                        l.best_overlap = best_ov
                        best_x = rect_list[best_idx]
                        best_y = rect_list[best_idx+1]
                        best_w = rect_list[best_idx+2]
                        best_h = rect_list[best_idx+3]
                        cv2.rectangle(img_left, (best_x, best_y), (best_x + best_w, best_y + best_h), (0,255,0), 2)
            # show image
            cv2.imshow("Results DSW", img_left)
            cv2.waitKey(0)
           

