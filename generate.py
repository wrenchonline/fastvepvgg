import time
import tensorflow as tf
from tensorflow.python.keras.engine.training import Model
import repvgg as rg
import datasets.datasets as da
batch_size= 1

def main():
    model = rg.get_RepVGG_func_by_name('RepVGG-A0')(deploy=False)
    model.build((None, 720, 1280, 3))
    checkpoint = tf.train.Checkpoint(myModel=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoint', max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    m_deploy = rg.repvgg_model_convert(model, 'RepVGG-A0',(None, 720, 1280, 3))
    m_deploy.summary()
    train_dataset = da.train_dataset.batch(batch_size)
    for step, (qf30 ,raw) in enumerate(train_dataset):
        y_pred = m_deploy(qf30)
        psnr = tf.image.psnr(y_pred, raw ,max_val=1.0).numpy()
        print('testing step %s: PSNR(图像质量):%s' % (step,psnr[0]))
        break
    converter = tf.lite.TFLiteConverter.from_keras_model(m_deploy)
    tflite_model = converter.convert()
    #'model.tflite'是输出的文件名
    savepath = r'model.tflite'
    open(savepath, "wb").write(tflite_model)
if __name__ == '__main__':
  main()