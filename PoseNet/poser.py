import cv2
import numpy as np

protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

frame = cv2.imread("unni.jpg")
frameCopy = np.copy(frame)
img_width = frame.shape[1]
img_height = frame.shape[0]
threshold = 0.1

input_blob = cv2.dnn.blobFromImage(frame,1.0/255,(640,480),(0,0,0),swapRB=False,crop=False)
 
net.setInput(input_blob)

output = net.forward()

H = output.shape[2]
W = output.shape[3]
points = []
for i in range(15):
    prob_map = output[0, i, :, :]
    minVal,prob,minLoc,point = cv2.minMaxLoc(prob_map)
 
    x = (img_width*point[0])//W
    y = (img_height*point[1])//H
 
    if prob > threshold :
        cv2.circle(frame,(x,y),5,(0,255,255),thickness=-1,lineType=cv2.FILLED)
        #cv2.putText(frame,"{}".format(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,0,255),3,lineType=cv2.LINE_AA)
        points.append((x,y))
    else :
        points.append(None)

for i in POSE_PAIRS:
    A = i[0]
    B = i[1]
    if points[A] and points[B]:
        cv2.line(frame,points[A],points[B],(255,0,0),3)
        cv2.circle(frame,points[A],8,(255,0,0),thickness=-1,lineType=cv2.FILLED)

cv2.imshow("Output",frame)
cv2.imwrite("output.jpg",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
