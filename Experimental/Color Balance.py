import cv2 as cv
import numpy as np
import os

img_names = ['LEFT', 'CENTER', 'RIGHT']

def detect_features(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    detector      = cv.xfeatures2d.SIFT_create()
    kps, features = detector.detectAndCompute(img, None)

    kps = np.float32([kp.pt for kp in kps])
    return kps, features

def match_keypoints(kps1, kps2, features1, features2):
    matcher     = cv.DescriptorMatcher_create('BruteForce')
    raw_matches = matcher.knnMatch(features1, features2, k=2)

    matches = []
    for match in raw_matches:
        if len(match) == 2 and match[0].distance < match[1].distance * 0.75:
            matches.append((match[0].trainIdx, match[0].queryIdx))

    pts1 = np.float32([kps1[i] for (_, i) in matches])
    pts2 = np.float32([kps2[i] for (i, _) in matches])

    return pts1, pts2

def find_homography(pts1, pts2):
    if len(pts1) > 4 and len(pts2) > 4:
        return cv.findHomography(pts1, pts2, cv.RANSAC, 4)
    return None

# Frames arranged by [left, center, right]
def find_homographies(frames):
    kps, features = list(), list()
    for image in frames:
        img_kps, img_features = detect_features(image)
        kps.append(img_kps)
        features.append(img_features)

    center_kps_shifted = [[x, y] for [x, y] in kps[1]] # So the homography matrix maps to the center

    # Find the homography matrix for the left and center portion
    ptsL, ptsC = match_keypoints(kps[0], center_kps_shifted,
                                 features[0], features[1])
    M          = find_homography(ptsL, ptsC)

    if M == None:
        raise Exception()
    leftH, status = M

    # Find the homography matrix for the center and right portion
    ptsC, ptsR = match_keypoints(center_kps_shifted, kps[2],
                                 features[1], features[2])
    M          = find_homography(ptsC, ptsR)

    if M == None:
        raise Exception()
    H, status = M
    rightH = np.linalg.inv(H)

    return leftH, rightH

def find_color_balance(frames):
    # Find homography
    kps, features = list(), list()
    for image in frames:
        img_kps, img_features = detect_features(image)
        kps.append(img_kps)
        features.append(img_features)

    center_kps_shifted = [[x, y] for [x, y] in kps[1]]

    # Find the homography matrix for the left and center portion
    ptsL, ptsC = match_keypoints(kps[0], center_kps_shifted,
                                 features[0], features[1])
    M = find_homography(ptsL, ptsC)

    if M == None:
        raise Exception()
    leftH, status = M

    # Find the homography matrix for the center and right portion
    ptsC, ptsR = match_keypoints(center_kps_shifted, kps[2],
                                 features[1], features[2])
    M = find_homography(ptsC, ptsR)

    if M == None:
        raise Exception()
    H, status = M
    rightH = np.linalg.inv(H)


    # Create mocks of the panoramas
    left = cv.warpPerspective(frames[0], leftH, (1920, 1080))
    right = cv.warpPerspective(frames[2], rightH, (1920, 1080))

    # Find overlap (left)
    left_active = (left != [0, 0, 0]).all(axis=2)
    frame_left = cv.bitwise_and(frames[1], frames[1], mask=left_active.astype(np.uint8))
    diff_left = frame_left.astype(np.int) - left.astype(np.int)
    diff_left = diff_left[np.where(left_active)]

    left_weights = list()
    for i in range(3):
        left_weights.append(int(np.average(diff_left[:, i])))

    # Find overlap (right)
    right_active = (right != [0, 0, 0]).all(axis=2).astype(np.uint8)
    frame_right = cv.bitwise_and(frames[1], frames[1], mask=right_active.astype(np.uint8))
    diff_right = frame_right.astype(np.int) - right.astype(np.int)
    diff_right = diff_right[np.where(right_active)]

    right_weights = list()
    for i in range(3):
        right_weights.append(int(np.average(diff_right[:, i])))

    return [left_weights, right_weights]


def balance_color(frames, weights):
    left = frames[0].astype(np.int)
    left[:, :, 0] += weights[0][0]
    left[:, :, 1] += weights[0][1]
    left[:, :, 2] += weights[0][2]

    left[left < 0]   = 0
    left[left > 255] = 255
    left                  = left.astype(np.uint8)

    right = frames[2].astype(np.int)
    right[:, :, 0] += weights[1][0]
    right[:, :, 1] += weights[1][1]
    right[:, :, 2] += weights[1][2]

    right[right < 0]   = 0
    right[right > 255] = 255
    right              = right.astype(np.uint8)

    return [left, frames[1], right]



frames = [cv.imread(os.path.join(os.getcwd(), 'Calib', '{}.jpg'.format(img_names[i])))
            for i in range(3)]

weights = find_color_balance(frames)
frames  = balance_color(frames, weights)


# leftH, rightH = find_homographies(frames)
# left          = cv.warpPerspective(frames[0], leftH, (1920, 1080))
# right         = cv.warpPerspective(frames[2], rightH, (1920, 1080))
#
# left_active = (left != [0, 0, 0]).all(axis=2)
# frame_left  = cv.bitwise_and(frames[1], frames[1], mask=left_active.astype(np.uint8))
# diff_left   = frame_left.astype(np.int) - left.astype(np.int)
# diff_left   = diff_left[np.where(left_active)]
#
# left_weights = list()
# for i in range(3):
#     left_weights.append(int(np.average(diff_left[:, i])))
#
# new_left = np.copy(frames[0]).astype(np.int)
# for i in range(3):
#     new_left[:, :, i] += left_weights[i]
#
# new_left[new_left < 0] = 0
# new_left[new_left > 255] = 255
# new_left = new_left.astype(np.uint8)
#
# right_active = (right != [0, 0, 0]).all(axis=2).astype(np.uint8)
# frame_right  = cv.bitwise_and(frames[1], frames[1], mask=right_active.astype(np.uint8))
# diff_right   = frame_right.astype(np.int) - right.astype(np.int)
# diff_right   = diff_right[np.where(right_active)]
#
# right_weights = list()
# for i in range(3):
#     right_weights.append(int(np.average(diff_right[:, i])))
#
# new_right = np.copy(frames[2]).astype(np.int)
# for i in range(3):
#     new_right[:, :, i] += right_weights[i]
#
# new_right[new_right < 0] = 0
# new_right[new_right > 255] = 255
# new_right = new_right.astype(np.uint8)
#
# cv.imwrite(os.path.join(os.getcwd(), "Calib", "1-New Left.jpg"), new_left)
# cv.imwrite(os.path.join(os.getcwd(), "Calib", "1-New Right.jpg"), new_right)
#
# new_left  = cv.warpPerspective(new_left, leftH, (1920, 1080))
# new_right  = cv.warpPerspective(new_right, rightH, (1920, 1080))
#
# cv.imwrite(os.path.join(os.getcwd(), "Calib", "1-L Corrected.jpg"), new_left)
# cv.imwrite(os.path.join(os.getcwd(), "Calib", "1-R Corrected.jpg"), new_right)