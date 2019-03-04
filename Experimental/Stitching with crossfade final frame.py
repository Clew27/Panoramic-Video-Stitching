import cv2 as cv
import numpy as np

IMAGE_FILENAMES = ['left', 'center', 'right']
OVERLAP         = 10 # %

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

    center_kps_shifted = [[x + 2880, y] for [x, y] in kps[1]] # So the homography matrix maps to the center

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

def find_left_mirror_homography(frames):
    left_kps, left_features = detect_features(frames[0])
    center_kps, center_features = detect_features(frames[1])

    left_kps_mirrored   = [[1920 - x, y] for [x, y] in left_kps]
    center_kps_mirrored = [[7680 - (x + 2880), y ] for [x, y] in center_kps]


    # Find the homography matrix for the center and right portion
    ptsC, ptsML = match_keypoints(center_kps_mirrored, left_kps_mirrored,
                                 center_features, left_features)
    M          = find_homography(ptsC, ptsML)

    if M == None:
        raise Exception()
    H, status = M
    leftMH = np.linalg.inv(H)

    return leftMH

def find_right_mirror_homography(frames):
    center_kps, center_features = detect_features(frames[1])
    right_kps, right_features   = detect_features(frames[2])

    center_kps_mirrored = [[7680 - (x + 2880), y] for [x, y] in center_kps]
    right_kps_mirrored  = [[1920 - x, y] for [x, y] in right_kps]

    # Find the homography matrix for the mirrored right and center portion
    ptsMR, ptsC = match_keypoints(right_kps_mirrored, center_kps_mirrored,
                                  right_features, center_features)
    M           = find_homography(ptsMR, ptsC)

    if M == None:
        raise Exception()
    rightMH, status = M

    return rightMH

def generate_masks(frames, leftH, rightH):
    panorama           = cv.warpPerspective(frames[0], leftH, (7680, 1080))
    right_warped       = cv.warpPerspective(frames[2], rightH, (7680, 1080))
    panorama[:, 3840:] = right_warped[:, 3840:]
    center_frame       = np.zeros((1080, 7680, 3), dtype=np.uint8)
    center_frame[:, 2880:4800] = frames[1]

    mask_raw = np.zeros(panorama.shape[:2], dtype=np.uint8)
    mask_raw = cv.rectangle(mask_raw, (2880, 0), (int(2880 + 1920 * OVERLAP / 100), 1080), 255, -1)
    mask_raw = cv.rectangle(mask_raw, (4799, 0), (int(4799 - 1920 * OVERLAP / 100), 1080), 255, -1)

    mask_center = np.zeros(panorama.shape[:2], dtype=np.uint8)
    mask_center = cv.rectangle(mask_center, (2880, 0), (4799, 1079), 255, -1)

    mask_lr = (panorama != [0, 0, 0]).all(axis=2)
    mask_lr = np.where(mask_lr, 255, 0).astype(np.uint8)

    mask_overlap          = np.bitwise_and(mask_raw, mask_lr)
    mask_overlap          = np.bitwise_and(mask_overlap, mask_center)
    mask_overlap_inverted = np.bitwise_not(mask_overlap)
    mask_center_inverted  = np.bitwise_not(mask_center)

    return {
        'mask_overlap'          : mask_overlap,
        'mask_overlap_inverted' : mask_overlap_inverted,
        'mask_center_inverted'  : mask_center_inverted
    }

def mask_gradient(frame, mask_overlap, center=True):
    frame = frame.astype(np.float64)
    if center:
        mask_gradient_center(frame)
    else:
        mask_gradient_lr(frame)
    frame = cv.bitwise_and(frame, frame, mask=mask_overlap)
    return frame.astype(np.uint8)

def mask_gradient_center(frame):
    rectangle_size = int(1920 * OVERLAP / 100) + 1
    for i in range(rectangle_size):
        blend_strength = i / (rectangle_size  - 1)
        frame[:, 2880 + i] *= blend_strength
        frame[:, 4799 - i] *= blend_strength

def mask_gradient_lr(frame):
    rectangle_size = int(1920 * OVERLAP / 100) + 1
    for i in range(rectangle_size):
        blend_strength = 1 - i / (rectangle_size  - 1)
        frame[:, 2880 + i] *= blend_strength
        frame[:, 4799 - i] *= blend_strength

def stitch_panorama(frames, leftH, rightH, masks):
    panorama           = cv.warpPerspective(frames[0], leftH, (7680, 1080))
    right_warped       = cv.warpPerspective(frames[2], rightH, (7680, 1080))
    panorama[:, 3840:] = right_warped[:, 3840:]
    center_frame       = np.zeros((1080, 7680, 3), dtype=np.uint8)
    center_frame[:, 2880:4800] = frames[1]

    center_frame_masked = mask_gradient(center_frame, masks['mask_overlap'], center=True)
    lr_frame_masked     = mask_gradient(panorama, masks['mask_overlap'], center=False)
    combined            = cv.add(center_frame_masked, lr_frame_masked)

    panorama      = cv.bitwise_and(panorama, panorama, mask=masks['mask_center_inverted'])
    panorama      = cv.bitwise_and(panorama, panorama, mask=masks['mask_overlap_inverted'])
    c_not_overlap = cv.bitwise_and(center_frame, center_frame, mask=masks['mask_overlap_inverted'])

    panorama = cv.add(panorama, c_not_overlap)
    panorama = cv.add(panorama, combined)

    return panorama





frames = list()
for img_filename in IMAGE_FILENAMES:
    frame = cv.imread('{}.jpg'.format(img_filename))
    frames.append(frame)

leftH, rightH = find_homographies(frames)

masks    = generate_masks(frames, leftH, rightH)
panorama = stitch_panorama(frames, leftH, rightH, masks)
non_img_pos  = (panorama == [0, 0, 0]).all(axis = 2)
non_img_pos  = np.where(non_img_pos)
top_row_x    = non_img_pos[1][np.where(non_img_pos[0] == 0)]
bottom_row_x = non_img_pos[1][np.where(non_img_pos[0] == 1079)]

top_left_x     = top_row_x[top_row_x < 3840][-1]
top_right_x    = top_row_x[top_row_x >= 3840][0]
bottom_left_x  = bottom_row_x[bottom_row_x < 3840][-1]
bottom_right_x = bottom_row_x[bottom_row_x >= 3840][0]

cutoff_left  = top_left_x if top_left_x > bottom_left_x else bottom_left_x
cutoff_right = top_right_x if top_right_x < bottom_right_x else bottom_right_x

cv.imwrite("Stitched Panorama.jpg", panorama)
cv.imwrite("Stitched Panorama Cropped.jpg", panorama[:, cutoff_left + 1:cutoff_right])


