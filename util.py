from tensorflow import keras
import cv2
import numpy as np

def load_model():
    model = keras.models.load_model("mymodel.h5")
    return model 
 


def pre_process(image_file):
    
  faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  nparr = np.frombuffer(image_file,np.uint8)
  image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray,1.1,4)
  for x,y,w,h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Face Not Detected")
        return "Null"
    else:
        for (ex,ey,ew,eh) in facess:
            face_roi = roi_color[ey: ey+eh, ex:ex + ew]
    image = cv2.resize(face_roi, (224,224))
    image = np.expand_dims(image,axis=0)
    image = image/255.0


  return image



def post_process(prediction):
  print(prediction)
  emotion_dict = {0:"angry",1:"sad",2:"neutral"}
  output = np.argmax(prediction)
  res=emotion_dict[int(output)]
      
  return res