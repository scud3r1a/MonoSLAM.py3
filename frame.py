# Import system dependencies
import os

# Import dependencies
from constants import RANSAC_RESIDUAL_THRES, RANSAC_MAX_TRIALS
import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import ransac
from helpers import add_ones, poseRt, fundamentalToRt, normalize, EssentialMatrixTransform, myjet

# Set numpy print options
np.set_printoptions(suppress=True)

# Extract features (from image) function
def extractFeatures(img):
    # Instantiate ORB creator with OpenCV
    orb = cv2.ORB_create()
    # Get good features to track from current image as points
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
    # Extract key points from points
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    # Apply ORB
    kps, des = orb.compute(img, kps)
    # Return key points and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

# Match frames function
def match_frames(f1, f2):
    # Instantiate BFMatcher with OpenCV
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # Use BFMatcher with k-nearest neighbors matching to find matched points within two images
    matches = bf.knnMatch(f1.des, f2.des, k=2)
    # Lowe's ratio test
    ret = []
    idx1, idx2 = [], []
    idx1s, idx2s = set(), set()
    # Iterate over matches
    for m, n in matches:
        # Only use points within a certain range
        if m.distance < 0.75 * n.distance:
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]
            # Only use points within ORB distance equal to 32
            if m.distance < 32:
                # Append points
                if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    idx1s.add(m.queryIdx)
                    idx2s.add(m.trainIdx)
                    ret.append((p1, p2))

    # Assertions
    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))
    assert len(ret) >= 8

    # Convert lists to numpy arrays
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # Identify transformation matrix between to frames using RANSAC algorithm
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            EssentialMatrixTransform,
                            min_samples=8,
                            residual_threshold=RANSAC_RESIDUAL_THRES,
                            max_trials=RANSAC_MAX_TRIALS)
    # Print model inliers and results
    print("Matches:  %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))
    # Return model inliers and model parameters
    print("Model params", model.params)
    print("...fundamental to R-t", fundamentalToRt(model.params))
    return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)

class Frame(object):
    def __init__(self, mapp, img, K, pose=np.eye(4), tid=None, verts=None):
        self.K = np.array(K)
        self.pose = np.array(pose)
        if img is not None:
            self.h, self.w = img.shape[0:2]
            if verts is None:
                self.kpus, self.des = extractFeatures(img)
            else:
                assert len(verts) < 256
                self.kpus, self.des = verts, np.array(list(range(len(verts)))*32, np.uint8).reshape(32, len(verts)).T
            self.pts = [None]*len(self.kpus)
        else:
            # fill in later
            self.h, self.w = 0, 0
            self.kpus, self.des, self.pts = None, None, None
        self.id = tid if tid is not None else mapp.add_frame(self)

    def annotate(self, img):
        # paint annotations on the image
        for i1 in range(len(self.kpus)):
            u1, v1 = int(round(self.kpus[i1][0])), int(round(self.kpus[i1][1]))
            if self.pts[i1] is not None:
                if len(self.pts[i1].frames) >= 5:
                    cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
                else:
                    cv2.circle(img, (u1, v1), color=(0,128,0), radius=3)
                # draw the trail
                pts = []
                lfid = None
                for f, idx in zip(self.pts[i1].frames[-9:][::-1], self.pts[i1].idxs[-9:][::-1]):
                    if lfid is not None and lfid-1 != f.id:
                        break
                    pts.append(tuple(map(lambda x: int(round(x)), f.kpus[idx])))
                    lfid = f.id
                if len(pts) >= 2:
                    cv2.polylines(img, np.array([pts], dtype=np.int32), False, myjet[len(pts)]*255, thickness=1, lineType=16)
            else:
                cv2.circle(img, (u1, v1), color=(0,0,0), radius=3)
        return img


    # inverse of intrinsics matrix
    @property
    def Kinv(self):
        if not hasattr(self, '_Kinv'):
            self._Kinv = np.linalg.inv(self.K)
        return self._Kinv

    # normalized keypoints
    @property
    def kps(self):
        if not hasattr(self, '_kps'):
            self._kps = normalize(self.Kinv, self.kpus)
        return self._kps

    # KD tree of unnormalized keypoints
    @property
    def kd(self):
        if not hasattr(self, '_kd'):
            self._kd = cKDTree(self.kpus)
        return self._kd