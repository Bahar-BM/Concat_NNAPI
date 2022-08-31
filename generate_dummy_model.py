#!/usr/bin/env python3
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Dense
from tensorflow.keras.models import Model

######## TEST - crash pattern - concatenation right after inputs  ########
x00 = Input(shape=(128, 128, 3))
x01 = Input(shape=(128, 128, 6))

x = Concatenate()([x00, x01])
x = Conv2D( 64, (3, 3), padding='same', use_bias=True)(x)

model = Model([x00, x01], [x], name='test')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('quantize_concat.h5')
model.save('quantize_concat')

####################### TEST Glove - TEST DENSE #############################
x0 = Input(shape=(4096))

x = Dense(1024)(x0)

model = Model([x0], [x], name='test')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('large_Dense.h5')
model.save('large_Dense')