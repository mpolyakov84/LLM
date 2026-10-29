import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import  load_model


model = load_model('model_v1_0922_1510.h5')


def cast_img(img):

    # img = tf.io.read_file(path)
    # img = tf.io.decode_jpeg(img, channels=3)
    img = tf.convert_to_tensor(img, dtype=tf.int32)
    img = tf.cast(img, tf.float32) / 255
    img = tf.image.resize(img, [480, 640])
    img = tf.expand_dims(img, axis=0)

    return img

def tf_model_prediction(img):
    img = cast_img(img)
    coord = model.predict(img)
    x_center, y_center, width, height = coord[0]

    return {'x_center': x_center, 'y_center': y_center, 'width': width, 'height': height}