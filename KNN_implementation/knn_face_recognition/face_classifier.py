#recognising faces using KNN
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))


def knn(X,Y,querypoint,k=5):
    val=[]
    m=X.shape[0]
    
    for i in range(m):
        d=dist(querypoint,X[i])
        val.append([d,Y[i]])
        
    val=sorted(val)
    #first k points
    val=val[:5]
    
    val=np.array(val)
    new_val=np.unique(val[:,1],return_counts=True)
#     print(new_val)
    index=new_val[1].argmax()
    prediction=new_val[0][index]
    return prediction

    #*********************************************************

import numpy as np
import cv2
import os
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
skip=0
face_data=[]
dataset_path="./data/"
label=[]
class_id=0
names={}

#data prepration
for fx in os.listdir(dataset_path):
    if fx.endswith(".npy"):
        names[class_id]=fx[:-4]
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        #create label for class
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        label.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_label=np.concatenate(label,axis=0).reshape((-1,1))
# print(face_dataset.shape)
# print(face_label.shape)
trainset=np.concatenate((face_dataset,face_label),axis=1)
# print(trainset.shape)

# dataset is ready to use now

# testing

while True:
    ret,frame=cap.read()
    if ret==False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x,y,w,h) in faces[-1:]:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2) 

        #crop out the required face
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        output=knn(face_dataset,face_label,face_section.flatten())
        pred_name=names[int(output)]
        #display name and rectange around it

        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

    cv2.imshow("face",frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
