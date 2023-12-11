import cv2 #opencv libraries for image processing
import datetime
import imutils # image processing
import numpy as np
from nms import non_max_suppression_fast
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.models import load_model
from centroidtracker import CentroidTracker
#facenet = FaceNet()

#from datetime import datetime


import pandas as pd

facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))


tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
def main():
    #cap = cv2.VideoCapture('test_video.mp4')
    cap=cv2.VideoCapture(0)
    #fps_start_time = datetime.datetime.now()
    #fps = 0
    
    #tmp=[]
    label=''

    # total_frames = 0
    # lpc_count = 0
    # opc_count = 0
    #dwell_time = dict()
    #dtime = dict()
    object_id_list = []
    #data=['id','intime','outtime']
    my_dict = {"Id":[],"In_time":[],"Gender":[]}
    #ret, frame = cap.read()
    #frame = imutils.resize(frame, width=600)

    while True:
        ret, frame = cap.read() # read frames from video or camera
        
        frame = imutils.resize(frame,width=600) # image resize to 600
        #frame = cv2.resize(frame, 600, cv2.INTER_AREA)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
 
 
        #get_hour=datetime.datetime.now()
    
        new_rects = []
        for x,y,w,h in faces:
            img = rgb_img[y:y+h, x:x+w]		         
            new_rects.append((x, y, x + w, y + h))	    
            img = cv2.resize(img, (160,160)) # 1x160x160x3		
            img = np.expand_dims(img,axis=0)		
            ypred = facenet.embeddings(img)		
            face_name = model.predict(ypred)	
            ypred = facenet.embeddings(img)	
            face_name = model.predict(ypred)
            ypred = facenet.embeddings(img)	
            face_name = model.predict(ypred)	
            final_name = encoder.inverse_transform(face_name)[0]	
            	
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 5)				
            cv2.putText(frame, str(final_name), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)

        
        
        boundingboxes = np.array(new_rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(new_rects)
        # for (objectId, bbox) in objects.items():
        #     x1, y1, x2, y2 = bbox
        #     x1 = int(x1)
        #     y1 = int(y1)
        #     x2 = int(x2)
        #     y2 = int(y2)

            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #text = "ID: {}".format(objectId)
            #cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            
            #if objectId not in object_id_list:
            #   object_id_list.append(objectId)
        for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
                text = "ID {}".format(objectID)
               

                if objectID not in object_id_list:		        
                        object_id_list.append(objectID)		        
                        now=datetime.datetime.now()                      
                        #dtime[objectID] = datetime.datetime.now()	                       
                        #dwell_time[objectID] = 0	        
                        # lock=0		        
                        # tmp.append(0)		        
                        time = now.strftime("%y-%m-%d %H:%M:%S")	        
                        #print(type(objectId))		        
                        my_dict["Id"].append((str(objectID)))		        
                        my_dict["In_time"].append(str(time))		        
                        my_dict["Gender"].append(str(final_name))
          
                   
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv2.imshow("Application", frame)
        #frame1 = frame2
        key = cv2.waitKey(1)
        if key == ord('q'):
            print(my_dict)
 
            df=pd.DataFrame.from_dict(my_dict)
            #df.set_index('In_time', inplace=True)
            #print(df[df["In_time"]< "16:46:00"])
            df.to_csv('person_face_gender.csv', index=False)   
            #df=df.loc[df['In_time']]
            #df=df[(df.index.hour>13)]
            #print(df)
            #ydf.loc[df['In_time'].dt.time > time(17,00)]
            break

    cv2.destroyAllWindows()


main()

