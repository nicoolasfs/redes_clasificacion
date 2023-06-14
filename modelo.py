import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.utils import to_categorical


num_classes=10
batch_size = 32
height = 200
width = 200
channels = 1

#Cargar los datos desde el archivo CSV
#Datos de entrenamiento
data_train = pd.read_csv('E:/Usuario/Proyecto de grado/Redes/dataset_train.csv')

#Datos de validación
data_val = pd.read_csv('E:/Usuario/Proyecto de grado/Redes/dataset_val.csv')

# Obtener las rutas de las imágenes y las etiquetas de entrenamiento
image_paths = data_train['image_path'].tolist()
data_train['category'] = data_train['category'].astype(str)

labels = data_train['category']
labels = to_categorical(labels, num_classes)

# Obtener las rutas de las imágenes y las etiquetas de validación
image_paths_val = data_val['image_path'].tolist()
data_val['category'] = data_val['category'].astype(str)


#Para visualizar la primera imagen
#Lo que significa que está correctamente mapeada la dirección de la imagen con el dataset origen
image_path = data_train['image_path'].iloc[0]
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis('off')
plt.show()

# Insertar las imagenes en un array, redimensionarlas y convertirlas a escala de grises
images = np.array([cv2.imread(path) for path in image_paths])
images = np.array([cv2.resize(image, (height, width)) for image in images])
images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images])
images = images.reshape(images.shape[0], height, width, channels)
images = images.astype('float32') / 255.0


#Mostrar todas las imagenes en una grilla
fig = plt.figure(figsize=(20, 20))
for i in range(1, 10):
    fig.add_subplot(3, 3, i)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
    
plt.show()

# Crear generador de datos para cargar y preprocesar las imágenes
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range = 50,             #Rotación aleatoria de las imagenes
    width_shift_range = 0.3,        #Desplazamiento aleatorio de las imagenes
    height_shift_range = 0.3,     
    shear_range = 15,              #Corte aleatorio de las imagenes
    zoom_range = [0.5, 1.5],      #Zoom aleatorio de las imagenes
    horizontal_flip = True,
    vertical_flip = True
    )  

val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 50,             #Rotación aleatoria de las imagenes
    width_shift_range = 0.3,        #Desplazamiento aleatorio de las imagenes
    height_shift_range = 0.3,     
    shear_range = 15,              #Corte aleatorio de las imagenes
    zoom_range = [0.5, 1.5],      #Zoom aleatorio de las imagenes
    horizontal_flip = True,
    vertical_flip = True
    )

# Configurar el generador de datos para cargar las imágenes y las etiquetas
print(data_train.head())

train_generator = train_datagen.flow_from_dataframe(
    dataframe=data_train,
    x_col='image_path',
    y_col='category',
    target_size=(200, 200),  # Dimensiones de las imágenes
    batch_size=batch_size,
    class_mode='sparse', # Tipo de problema de clasificación
    classes=[str(i) for i in range(num_classes)],
    color_mode='grayscale'
)

test_generator = val_datagen.flow_from_dataframe(
    dataframe=data_val,
    x_col='image_path',
    y_col='category',
    target_size=(200, 200),  # Dimensiones de las imágenes
    batch_size=batch_size,
    class_mode='sparse', # Tipo de problema de clasificación
    classes=[str(i) for i in range(num_classes)],
    color_mode='grayscale'
)

# Crear el modelo de red neuronal convolucional
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print ("Modelo compilado")

#Observacion y entrenamiento de las redes
#Para observar: tensorboard --logdir="E:/Usuario/Proyecto de grado/Redes/Dataset/Board"
# Entrenar el modelo
board = TensorBoard(log_dir='E:/Usuario/Proyecto de grado/Redes/Dataset/Board/CNN')
model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[board])
print ("Modelo entrenado")


# Guardar el modelo entrenado
model.save('model.h5')
model.save_weights('model_weights.h5')
print ("Modelo guardado")
