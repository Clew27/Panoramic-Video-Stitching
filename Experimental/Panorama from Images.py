import numpy as np
import cv2 as cv

### FUNCTIONS
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

def draw_features(img, kps):
    final = np.zeros(img.shape[:2])
    for kp in kps:
        final = cv.circle(img       = img,
                          center    = (kp[0], kp[1]),
                          radius    = 3,
                          color     = (0, 0, 255),
                          thickness = -1)
    return final

def resize(img, factor):
    size = (int(img.shape[1] / factor), int(img.shape[0] / factor))
    return cv.resize(img, size)

### MAIN CODE
images = list()
images.append(cv.imread('LEFT.jpg'))
images.append(cv.imread('CENTER.jpg'))
images.append(cv.imread('RIGHT.jpg'))

kps, features = list(), list()
for image in images:
    img_kps, img_features = detect_features(image)
    kps.append(img_kps)
    features.append(img_features)

center_kps_shifted = [[x + 2880, y + 540] for [x, y] in kps[1]]

ptsL, ptsC = match_keypoints(kps[0], center_kps_shifted,
                             features[0], features[1])
M          = find_homography(ptsL, ptsC)

if M == None:
    print("Left stitch error")
H, status = M

warp_left = cv.warpPerspective(images[0], H, (7680, 2160))



ptsC, ptsR = match_keypoints(center_kps_shifted, kps[2],
                             features[1], features[2])
M          = find_homography(ptsC, ptsR)

if M == None:
    print("Right stitch error")
H, status = M
inv_h = np.linalg.inv(H)

warp_right = cv.warpPerspective(images[2], inv_h, (7680, 2160))

warp_left[:, 3840:] = warp_right[:, 3840:]

# height, width = warp_left.shape[:2]
# for r in range(height):
#     for c in range(width):
#         if not np.array_equal(warp_right[r, c], [0, 0, 0]):
#             warp_left[r, c] = warp_right[r, c]

warp_left[540:1620, 2880:4800] = images[1]
warp_left = resize(warp_left, 3)
cv.imshow("Panorama", warp_left)
cv.imwrite("Panorama.jpg", warp_left)

# warp_left[540:1620, 2880:4800] = images[1]
# warp_right[540:1620, 2880:4800]  = images[1]
#
# warp_left = resize(warp_left, 3)
# warp_right = resize(warp_right, 3)
# cv.imshow("Pano left", warp_left)
# cv.imshow("Pano right", warp_right)

cv.waitKey()
cv.destroyAllWindows()

