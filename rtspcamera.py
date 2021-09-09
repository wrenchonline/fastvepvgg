import time
import tensorflow as tf

from tensorflow.python.keras.engine.training import Model
import repvgg as rg
import datasets.datasets as da
import cv2

batch_size= 1

model = rg.get_RepVGG_func_by_name('RepVGG-A0')(deploy=False)
model.build((None, 720, 1280, 3))
checkpoint = tf.train.Checkpoint(myModel=model)
manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoint', max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)
m_deploy = rg.repvgg_model_convert(model, 'RepVGG-A0',(None, 720, 1280, 3))
m_deploy.summary()

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [720, 1280])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


image_raw_data = load_and_preprocess_image('data\\qf30\\1.jpeg')
image_raw_data = tf.expand_dims(image_raw_data, 0)
# image_data = tf.image.decode_jpeg(image_raw_data)
# img_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)

start2 = time.time()
tensor_y = m_deploy(image_raw_data)
tensor_y = tf.squeeze(tensor_y)
img_data = tf.image.convert_image_dtype(tensor_y, dtype=tf.uint8)
img_data = img_data.numpy()
end2= time.time()
print("Model Elapsed (with compilation) = %s" % (end2 - start2))
# cv2.imwrite('qf30.jpeg', img_data,[int(cv2.IMWRITE_JPEG_QUALITY), 50])

cv2.imwrite('qf30.png', img_data)


