import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
skip=0
face_data=[]
dataset_path="./data/"
file_name=input("enter name : ")
while True:
    ret,frame=cap.read()
    if ret==False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("frame",frame)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    #pick the last face (last face will be the largest acc to area)
    for (x,y,w,h) in faces[-1:]:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2) 

        #crop out the required face
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        skip+=1
        
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))


    cv2.imshow("frame",frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+".npy",face_data)
print("data successfully saved at "+dataset_path+file_name+".npy")
cap.release()
cv2.destroyAllWindows()
