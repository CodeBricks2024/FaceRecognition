# import coremltools
# import tensorflow as tf
#
# tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
#
# if tf_version == 1:
#     from tensorflow.keras.models import Model, Sequential
#     from tensorflow.keras.layers import (
#         Convolution2D,
#         ZeroPadding2D,
#         MaxPooling2D,
#         Flatten,
#         Dropout,
#         Activation,
#     )
# else:
#     from tensorflow.keras.models import Model, Sequential
#     from tensorflow.keras.layers import (
#         Convolution2D,
#         ZeroPadding2D,
#         MaxPooling2D,
#         Flatten,
#         Dropout,
#         Activation,
#     )
#
# # 다운 받아놓은 weights의 경로
# weights_path = './vgg_face_weights.h5'
#
# # Model 구성
# model = Sequential()
# model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
# model.add(Convolution2D(64, (3, 3), activation="relu"))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(64, (3, 3), activation="relu"))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, (3, 3), activation="relu"))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, (3, 3), activation="relu"))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, (3, 3), activation="relu"))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, (3, 3), activation="relu"))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, (3, 3), activation="relu"))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, (3, 3), activation="relu"))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, (3, 3), activation="relu"))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, (3, 3), activation="relu"))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, (3, 3), activation="relu"))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, (3, 3), activation="relu"))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, (3, 3), activation="relu"))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(Convolution2D(4096, (7, 7), activation="relu"))
# model.add(Dropout(0.5))
# model.add(Convolution2D(4096, (1, 1), activation="relu"))
# model.add(Dropout(0.5))
# model.add(Convolution2D(2622, (1, 1)))
# model.add(Flatten())
# model.add(Activation("softmax"))
#
# # 모델에 다운 받아놓은 가중치 추가
# model.load_weights(weights_path)
#
#
# # Model을 .mlmodel 확장자로 변환
#
# converted_model = coremltools.convert(model, convert_to="mlprogram")
# # converted_model = coremltools.convert(model, convert_to="ImageClassifier")
# converted_model.save("ConvertedModel")
