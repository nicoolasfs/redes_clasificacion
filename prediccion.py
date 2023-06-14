from re import T
import tensorflow as tf
import cv2
import numpy as np
from keras_preprocessing.image import img_to_array

#Direcciones de los modelos tras entrenar
ModeloDenso = 'C:/Users/nfons/Documents/Universidad/Proyecto de grado/Redes/'
ModeloConvolucional = 'C:/Users/nfons/Documents/Universidad/Proyecto de grado/Redes/'
ModeloCNN2 = 'C:/Users/nfons/Documents/Universidad/Proyecto de grado/Redes/'

#Se leen las redes neuronales
#Red densa
Denso = tf.keras.models.load_model(ModeloDenso)
pesosDenso = Denso.get_weights()
Denso.set_weights(pesosDenso)

#Red convolucional
Convolucional = tf.keras.models.load_model(ModeloConvolucional)
pesosConvolucional = Convolucional.get_weights()
Convolucional.set_weights(pesosConvolucional)

#Red CNN2
CNNdrop = tf.keras.models.load_model(ModeloCNN2)
pesosCNN2 = CNNdrop.get_weights()
CNNdrop.set_weights(pesosCNN2)

#Esta funcion se encarga de capturar la imagen de la camara
cap = cv2.VideoCapture(1) 

while True:
    #Lectura de la videocaptura
    ret, frame = cap.read()

    #pasar la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Se redimensiona la imagen 
    gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_CUBIC)

    #se normaliza la imagen
    gray = np.array(gray).astype(float)/255

    #se convierte la imagen en una matriz
    img = img_to_array(gray)
    img = np.expand_dims(img, axis=0)

    #Se realiza la prediccion
    prediccion = CNNdrop.predict(img)
    prediccion = prediccion[0]
    prediccion = prediccion[0]
    print(prediccion)

    #Clasificacion
    if prediccion <= 0.5:
        cv2.putText(frame, "Caballo", (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Humano", (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #Se muestran los fotogramas
    cv2.imshow('CNNdrop', frame)

    t= cv2.waitKey(1)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()

