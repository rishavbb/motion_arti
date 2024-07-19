import os
import cv2
import imutils
import numpy as np


assert len(os.listdir("data/raw")) == len(os.listdir("data/annotated")),\
       "Same file names should exist in data/raw and data/annotated"
       

H = 50


def align_images(image, template, maxFeatures=500, keepPercent=0.2,
	debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)


    matches = sorted(matches, key=lambda x:x.distance)

    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
            matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)


    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    for (i, m) in enumerate(matches):

        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt


    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    return aligned


img_paths = os.listdir("data/raw")


starting_h = 0

for img_name in img_paths:
    img1 = cv2.imread("data/raw/"+img_name)[0:410, 0:410]
    img2 = cv2.imread("data/annotated/"+img_name)[0:410, 0:410]

    aligned = align_images(img1, img2, debug=False)



    # for starting_h
    for starting_h in [360, 310, 260, 210, 160, 110, 60, 10, 0]:
        for i in range(3):
            patch1 = aligned[starting_h:starting_h+H, :]
            patch2 = img2[starting_h:starting_h+H, :]
            img_name_ = img_name.split(".")[0]
            cv2.imwrite("data/train/perfect/"+img_name_+"_"+str(starting_h)+".jpg", patch1)


            mask = np.zeros_like(patch1)
            glitch_height = np.random.randint(3, 9)

            random_glitch_id = np.random.randint(0+glitch_height, H-glitch_height)
            mask[random_glitch_id:random_glitch_id+glitch_height] = 1

            if np.random.randint(0, 3) == 0:
                random_glitch_id = np.random.randint(0, H-glitch_height)
                mask[random_glitch_id:random_glitch_id+glitch_height] = 1

            patch2 = patch2 * mask

            anti_mask = np.where(mask==1, 0, 1)
            patch1 = patch1 * anti_mask


            final_img = patch1 + patch2

            cv2.imwrite("data/train/tampered/"+img_name_+"_"+str(starting_h)+".jpg", final_img)

            blur_tamp = cv2.cvtColor(np.float32(final_img), cv2.COLOR_BGR2GRAY)
            blur_tamp = cv2.GaussianBlur(np.float32(blur_tamp), (5, 5), 0)
            ret, otsu = cv2.threshold(blur_tamp.astype("uint8"), 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            otsu = np.where(otsu==255, 1, 0)

            mask = mask * otsu[..., None]
            cv2.imwrite("data/train/mask/"+img_name_+"_"+str(starting_h)+".jpg", mask*255)
