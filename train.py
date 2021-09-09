import datasets.datasets as da
import tensorflow as tf
import repvgg as rg


batch_size = 1
iters = int(8e+5)

model = rg.get_RepVGG_func_by_name('RepVGG-A0')(deploy=False)
model.build((None, 720, 1280, 3))
model.summary()
#优化器.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#定义自己的损失函数,对比图片使用均值方差
loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(qf30,raw):
    with tf.GradientTape() as tape:    
        y_pred = model(qf30)
        loss_value = loss_fn(raw, y_pred) #压缩的图像与原图进行均值比较，会把压缩图像质量增大
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value , raw , y_pred


def main():
    checkpoint = tf.train.Checkpoint(myModel=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoint', max_to_keep=3)
    number = 0
    if manager.latest_checkpoint:
        number = int(manager.latest_checkpoint.split("-")[1])
    print(manager.latest_checkpoint)
    checkpoint.restore(manager.latest_checkpoint)
    train_dataset = da.train_dataset.batch(batch_size)
    epoches =int((iters - number) // batch_size) 
    # 使输入流水线可以在模型训练时异步获取批处理。
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    myiter = number
    for epoch in range(epoches):
        print('epoch: ', epoch)
        for step, (qf30 ,raw) in enumerate(train_dataset):
            loss_value , raw_frame , y_pred = train_step(qf30,raw)
            if step != 0 and step % 100 == 0:
                myiter += 100
                psnr = tf.image.psnr(y_pred, raw_frame ,max_val=1.0).numpy()
                print('Training loss (for one batch) at step %s: Loss:%s PSNR(图像质量):%s' % (myiter, float(loss_value),psnr[0]))
                manager.save(checkpoint_number=myiter)
                
if __name__ == '__main__':
  main()
    
    
    