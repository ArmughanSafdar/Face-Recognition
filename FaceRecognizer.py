import cv2
import numpy as np
import face_recognition
import os                                    #to find files from path and encodes them automatically

path= 'ImagesFaceRecognizer'                 #recognizes the images only in this folder'ImagesFaceRecognizer'
images =[]                                   #creating list to get images in path
classNames = []                              #creating another list to get names in path(Empty bcz we can get file names)
my_list = os.listdir(path)                   #for grabbing the images in path(folder)
print(my_list)

for CLASS in my_list:
    current_Image = cv2.imread(f'{path}/{CLASS}')
    images.append(current_Image)
    classNames.append(os.path.splitext(CLASS)[0])          #to remove .png from our list
print(classNames)

def findEncodings(images):                                  #defining function
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)           #for coverting into RGB(As we have in BGR but library understands it as RGB)
        encode = face_recognition.face_encodings(img)[0]    #for finding the encoded list
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)                     #calling defined function
print(len(encodeListKnown))                                 #to show how amny time encoding have done
print("Encoding Completes")

cap = cv2.VideoCapture(0)                                   #for using webcam to capture image

while True:                                                 #Using While loop to get each frame one by one
    success, img = cap.read()                               #this will give us image
    small_img = cv2.resize(img,(0,0),None,0.25,0.25)   #resizing image (will help us speeding the process) as we have to run on real-time, it becomes 1/4th
    small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)  #RGB converter

    faces_in_current_frame = face_recognition.face_locations(small_img)         #for finding face in webcam
    encoding_in_current_frame = face_recognition.face_encodings(small_img,faces_in_current_frame) #to find encoding in webcam

    for encodeFace, face_location in zip(encoding_in_current_frame,faces_in_current_frame): #zip is used as we need both faces & encodings in current frame in the same loop
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)    #for finding match of webcam with given images
        face_distance = face_recognition.face_distance(encodeListKnown,encodeFace)    #it will return us a list with distances
        print(face_distance)                                #prints list with distance so lowest will be our best match
        matchIndex = np.argmin(face_distance)               #to detect lowest distance(argmin for minimum)

        if matches[matchIndex]:                            #if lowest then it prints name of that respective person
            name = classNames[matchIndex].upper()          #Name should be written in uppercase letters
            print(name)
            y1,x2,y2,x1 =face_location                                      #face-location detector
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4    #bcz image had become 1/4th so we multiply it with 4 to get it correctly
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)               #create rectangle around the face
            cv2.rectangle(img,(x1,y2-25),(x2,y2),(0,255,0),cv2.FILLED)   #create filled lower-end of formed rectangle
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),2) #shows the name of best match person

    cv2.imshow('Webcam',img)                          #shows the image on webcam
    cv2.waitKey(1)
