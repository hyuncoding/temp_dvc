import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import dvc.api
import os

IMAGE_SIZE = 224
BATCH_SIZE = 64

# 데이터 경로 설정
data_path = 'data/images'

# ImageDataGenerator 설정
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Xception 모델 설정
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
x = base_model.output

# 분류기
x = GlobalAveragePooling2D()(x)
x = Dropout(rate=0.5)(x)
x = Dense(100, activation='relu')(x)
x = Dropout(rate=0.5)(x)
output = Dense(23, activation='softmax', name='output')(x)

model = Model(inputs=input_tensor, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(train_generator, epochs=5, validation_data=validation_generator)

# 모델 저장
model.save('model/xception_model.h5')
