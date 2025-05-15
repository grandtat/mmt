import cv2
from os import listdir
from os.path import isfile, join

def compute_matchcount(file_path1, file_path2):
        img1 = cv2.imread(file_path1)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1_gray = cv2.blur(img1_gray,(5,5))
        img2 = cv2.imread(file_path2)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.blur(img2_gray,(5,5))
        # Feature Matching mit SIFT
        sift = cv2.SIFT_create()
        #sift = cv2.AKAZE_create()
        #sift = cv2.ORB_create(nfeatures=4000)
        kp1, desc1 = sift.detectAndCompute(img1_gray, None)
        kp2, desc2 = sift.detectAndCompute(img2_gray, None)
        
        # BFMatcher mit Hamming-Distanz (für binäre Features: ORB/BRIEF würden cv2.NORM_HAMMING nutzen)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.knnMatch(desc1, desc2, k=2)
        #matches = bf.match(desc1, desc2)
        
        # Lowe's Ratio Test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.55 * n.distance:
                good_matches.append(m)
        ##bis hier 
        return len(good_matches)
        #return len(matches)

if __name__ == "__main__":
    mypath = "c/"
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    d={}
    for file1 in onlyfiles:
        d1 ={}
        for file2 in onlyfiles:
            if not file1 == file2:
                d1[file2]=compute_matchcount(file1,file2)
        d[file1]=d1
    print(d)
    for file1 in d.keys():
         d1=d[file1]
         file2=max(d1, key=d1.get)
         print(file1+" --> "+file2)