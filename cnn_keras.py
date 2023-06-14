import os
import tensorflow as tf
import cv2 #pip install opencv-python - si es necesario cambiar el interprete (ctrl + shift + p))      
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten
#La libreria ImageDataGenerator se instala de aqui https://vijayabhaskar96.medium.com/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1
from keras_preprocessing.image import ImageDataGenerator

#Para los dataset consultar en Tensorflow datasets (https://www.tensorflow.org/datasets/catalog/overview)
#El data set de humans or horses se encuentra aqui https://laurencemoroney.com/datasets.html
#pip install SciPy 

#Dirección de imagenes para entrenamiento y validación
entrenamiento = 'C:/Users/nfons/Documents/Proyecto de grado/Redes/Dataset/Entrenamiento'
validacion = 'C:/Users/nfons/Documents/Proyecto de grado/Redes/Dataset/Validacion'

ListaTrain = os.listdir(entrenamiento)
ListaTest = os.listdir(validacion)

#Se establecen parametros
ancho, alto = 200, 200

#listas para entrenamiento
etiquetas = []
fotos = []
datos_train = []
con = 0

#Listas para validación
etiquetas2 = []
fotos2 = []
datos_vali=[]
con2 = 0

#Se extrae en una lista las fotos y etiquetas
#Entrenamiento
for nameDir in ListaTrain:
    nombre = entrenamiento + '/' + nameDir #se leen las fotos

    for fileName in os.listdir(nombre): #se asignan etiquetas a cada foto
        etiquetas.append(con) #valores de etiqueta, 0 a la primera y 1 a la segunda
        img = cv2.imread(nombre + '/' + fileName, 0) #se leen las fotos
        img = cv2.resize(img, (ancho, alto),    interpolation=cv2.INTER_CUBIC) #se redimensionan las fotos
        img = img.reshape(ancho, alto, 1) #se convierten en matrices
        datos_train.append([img, con]) #se agregan a la lista de datos
        fotos.append(img) #se agregan a la lista de fotos
    
    con += 1

#Validación
for nameDir2 in ListaTest:
    nombre2 = validacion + '/' + nameDir #se leen las fotos

    for fileName2 in os.listdir(nombre2): #se asignan etiquetas a cada foto
        etiquetas2.append(con2) #valores de etiqueta, 0 a la primera y 1 a la segunda
        img2 = cv2.imread(nombre2 + '/' + fileName2, 0) #se leen las fotos
        img2 = cv2.resize(img2, (ancho, alto),    interpolation=cv2.INTER_CUBIC) #se redimensionan las fotos
        img2 = img2.reshape(ancho, alto, 1) #se convierten en matrices
        datos_vali.append([img2, con2]) #se agregan a la lista de datos
        fotos2.append(img2) #se agregan a la lista de fotos
    
    con2 += 1
         
#Normalizacion de imagenes
#convertimos las imagenes en escala de grises
fotos = np.array(fotos).astype(float)/255
print (fotos.shape)
fotos2 = np.array(fotos2).astype(float)/255
print (fotos2.shape)

#Se convierten las etiquetas en un array
etiquetas = np.array(etiquetas)
etiquetas2 = np.array(etiquetas2)

imgTrainGen = ImageDataGenerator(
    rotation_range = 50,             #Rotación aleatoria de las imagenes
    width_shift_range = 0.3,        #Desplazamiento aleatorio de las imagenes
    height_shift_range = 0.3,     
    shear_range = 15,              #Corte aleatorio de las imagenes
    zoom_range = [0.5, 1.5],      #Zoom aleatorio de las imagenes
    horizontal_flip = True,
    vertical_flip = True,
)

imgTrainGen.fit(fotos)
plt.figure(figsize=(20,8))
for imagen, etiqueta in imgTrainGen.flow(fotos, etiquetas, batch_size=10, shuffle=False):
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[i], cmap='gray')
    plt.show()
    break

imgTrain = imgTrainGen.flow(fotos, etiquetas, batch_size=32)

#Red neuronal densa
#Esta red neuronal no tiene capas convolucionales, por ende trabaja mejor con problemas numericos más no con imagenes
ModeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(200, 200, 1)), #capa de entrada con 40000 neuronas
    tf.keras.layers.Dense(150, activation='relu'),    #capa oculta con 150 neuronas
    tf.keras.layers.Dense(150, activation='relu'),     #capa oculta con 150 neuronas
    tf.keras.layers.Dense(1, activation='sigmoid')    #capa de salida con 2 neuronas
])

#Red neuronal convolucional
#Esta red neuronal tiene capas convolucionales, por ende trabaja mejor con imagenes
ModeloConvolucional = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 1)), #capa de entrada con 32 kernels de 3x3
    tf.keras.layers.MaxPooling2D(2, 2), #capa de max pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), #capa oculta con 64 kernels de 3x3
    tf.keras.layers.MaxPooling2D(2,2), #capa de max pooling
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), #capa oculta con 128 kernels de 3x3
    tf.keras.layers.MaxPooling2D(2,2), #capa de max pooling

    #capas densas de clasificación
    tf.keras.layers.Flatten(), #capa de entrada con 20000 neuronas
    tf.keras.layers.Dense(256, activation='relu'), #capa oculta con 256 neuronas
    tf.keras.layers.Dense(1, activation='sigmoid') #capa de salida con 2 neuronas
])

#Red neuronal convolucional con dropout
#Esta red encuentra caminos distintos con cada entrenamiento, por ende es más robusta
ModeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 1)), #capa de entrada con 32 kernels de 3x3
    tf.keras.layers.MaxPooling2D(2, 2), #capa de max pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), #capa oculta con 64 kernels de 3x3
    tf.keras.layers.MaxPooling2D(2,2), #capa de max pooling
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), #capa oculta con 128 kernels de 3x3
    tf.keras.layers.MaxPooling2D(2,2), #capa de max pooling

    #capas densas de clasificación
    tf.keras.layers.Dropout(0.5), #capa de dropout
    tf.keras.layers.Flatten(), #capa de entrada con 20000 neuronas
    tf.keras.layers.Dense(256, activation='relu'), #capa oculta con 256 neuronas
    tf.keras.layers.Dense(1, activation='sigmoid') #capa de salida con 2 neuronas
])

#Se compilan los modelos
ModeloDenso.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ModeloConvolucional.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ModeloCNN2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Observacion y entrenamiento de las redes
#Para observar: tensorboard --logdir="C:/Users/nfons/Documents/Proyecto de grado/Redes/Dataset/Board"
#Entrenamiento de modelo denso
BoardDenso = TensorBoard(log_dir='C:/Users/nfons/Documents/Proyecto de grado/Redes/Dataset/Board/Denso')
ModeloDenso.fit(imgTrain, batch_size=32, epochs=100, validation_data=(fotos2, etiquetas2), callbacks=[BoardDenso], steps_per_epoch=int(np.ceil(len(fotos)/32)), 
            validation_steps=int(np.ceil(len(fotos2)/32)))
            
#Guardado del modelo denso
ModeloDenso.save('ClasificadorDenso.h5')
ModeloDenso.save_weights('PesosDenso.h5')
print ("Modelo denso guardado")

#Entrenamiento de modelo convolucional
BoardCNN = TensorBoard(log_dir='C:/Users/nfons/Documents/Proyecto de grado/Redes/Dataset/Board/CNN')
ModeloConvolucional.fit(imgTrain, batch_size=32, epochs=100, validation_data=(fotos2, etiquetas2), callbacks=[BoardCNN], steps_per_epoch=int(np.ceil(len(fotos)/32)), 
            validation_steps=int(np.ceil(len(fotos2)/32)))

#Guardado del modelo convolucional
ModeloConvolucional.save('ClasificadorCNN.h5')
ModeloConvolucional.save_weights('PesosCNN.h5')
print ("Modelo convolucional guardado")

#Entrenamiento de modelo convolucional con dropout
BoardCNN2 = TensorBoard(log_dir='C:/Users/nfons/Documents/Proyecto de grado/Redes/Dataset/Board/CNN2')
ModeloCNN2.fit(imgTrain, batch_size=32, epochs=100, validation_data=(fotos2, etiquetas2), callbacks=[BoardCNN2], steps_per_epoch=int(np.ceil(len(fotos)/32)), 
            validation_steps=int(np.ceil(len(fotos2)/32)))

#Guardado del modelo convolucional con dropout
ModeloCNN2.save('ClasificadorCNN2.h5')
ModeloCNN2.save_weights('PesosCNN2.h5')
print ("Modelo convolucional con dropout guardado") 