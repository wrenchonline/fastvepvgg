import tensorflow as tf
import os
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
from tensorflow.python.keras.backend import conv2d
 
class ConvBn(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides=1, padding='same', groups=1, **kwargs):
    super(ConvBn, self).__init__(**kwargs)
    self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                                       use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization()
    
  def call(self, inputs, training):
    x = inputs
    x = self.conv(x)
    x = self.bn(x, training=training)
    return x
 
class RepVGGBlock(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides=1, padding='same', groups=1, deploy=False, **kwargs):
    super(RepVGGBlock, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.groups = groups
    self.deploy = deploy
 
  def build(self, input_shape):
    self.in_channels = input_shape[-1]
    if self.deploy:
        # 部署
        self.rbr_reparam = tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding,
                                                  groups=self.groups,use_bias=True)
    else:
        # 训练
        self.rbr_identity = tf.keras.layers.BatchNormalization() if self.in_channels == self.filters and self.strides == 1 else None
        self.rbr_dense = ConvBn(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, groups=self.groups)
        self.rbr_1x1 = ConvBn(self.filters, kernel_size=1, strides=self.strides, padding=self.padding, groups=self.groups)
        tf.print('RepVGG Block, identity = ', self.rbr_identity)
    super(RepVGGBlock, self).build(input_shape)

  def call(self, inputs, training):
    x = inputs
    if hasattr(self, 'rbr_reparam'):
      return tf.nn.relu(self.rbr_reparam(x)) 
    if self.rbr_identity:
      id_out = self.rbr_identity(x, training=training)
    else:
      id_out = 0
    x = tf.nn.relu(self.rbr_dense(x, training=training) + self.rbr_1x1(x, training=training) + id_out)
    return x
  
  def get_equivalent_kernel_bias(self):
    '''计算等价kernel与bias'''
    kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
    kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
    kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
    return kernel3x3 + tf.pad(kernel1x1, [[1,1],[1,1],[0,0],[0,0]]) + kernelid, bias3x3 + bias1x1 + biasid
 
  def _fuse_bn_tensor(self, branch):
    if branch is None:
      return 0, 0
    if isinstance(branch, ConvBn):
      kernel = branch.conv.kernel
      moving_mean = branch.bn.moving_mean
      moving_variance = branch.bn.moving_variance
      gamma = branch.bn.gamma
      beta = branch.bn.beta
      eps = branch.bn.epsilon
    else:
      assert isinstance(branch, tf.keras.layers.BatchNormalization)
      if not hasattr(self, 'id_tensor'):
        input_dim = self.in_channels // self.groups
        kernel_value = tf.zeros((3, 3, input_dim, self.in_channels), dtype=tf.float32)
        for i in range(self.in_channels):
          kernel_value = tf.tensor_scatter_nd_update(kernel_value, [[1, 1, i % input_dim, i]], [1])
        self.id_tensor = kernel_value
      kernel = self.id_tensor
      moving_mean = branch.moving_mean
      moving_variance = branch.moving_variance
      gamma = branch.gamma
      beta = branch.beta
      eps = branch.epsilon
    std = tf.math.sqrt(moving_variance + eps)
    t = tf.reshape(gamma / std, (1, 1, 1, -1))
    return kernel * t, beta - moving_mean * gamma / std
 
  def repvgg_convert(self):
    kernel, bias = self.get_equivalent_kernel_bias()
    return kernel, bias


#（未完成）计算现有的编码器解码器预算输出的bit大小，以此来控制
class BENet(tf.keras.Model):  
    def __init__(self,**kwargs):
        self.conv0 = tf.keras.layers.Conv2D(filters=64,kernel_size=5,strides=2,padding='same')

        
#（未测试）此CNN网络模拟现有的编码器解码器        
class Acn_H264(tf.keras.Model):
  def __init__(self, **kwargs):
      pass
    
  #上采样  H264默认是8x8分块
  def up(self,factor, filters):
    net = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,padding='same')
    net = tf.nn.depth_to_space(net,block_size=factor) #因为是图片我这里直接选择这样去处理
    return net
  
  #下采样(压缩)  H264默认是8x8分块
  def down(self,factor, filters):
    net = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,padding='same')
    net = tf.nn.space_to_depth(net,block_size=factor) #因为是图片我这里直接选择这样去处理
    return net
  
  def build(self, input_shape=None):
    self.cur_layer_idx = 1
    self.up = self.up(filters=3,factor=8)
    self.conv0 = tf.keras.layers.Conv2D(filters=3, kernel_size=3,padding='same')
    self.down = self.down(filters=3,factor=8)
    
  def call(self,inputs):
    x = inputs
    x = self.up(x)
    x = self.conv0(x)
    x = tf.nn.relu(x)
    x = self.down(x)
    return x


 
 
class RepVGG(tf.keras.Model):
  def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, **kwargs):
    super(RepVGG, self).__init__(**kwargs)
    self.num_blocks = num_blocks
    self.num_classes = num_classes
    self.width_multiplier = width_multiplier
    self.override_groups_map = override_groups_map or dict()
    self.deploy = deploy
    assert len(self.width_multiplier) == 4
    assert 0 not in self.override_groups_map
    self.in_planes = min(64, int(64 * self.width_multiplier[0]))
 
  def build(self, input_shape):
    self.cur_layer_idx = 1
    self.conv0 = tf.keras.layers.Conv2D(filters=64,kernel_size=5,strides=2,padding='same') 
    #self.stage0 = RepVGGBlock(self.in_planes, kernel_size=3, strides=2, padding='same', deploy=self.deploy)
    self.stage1x1 = tf.keras.layers.Conv2D(filters=12,kernel_size=1,strides=1,padding='same') 
    self.stage1 = self._make_stage(12,3, stride=1)
    self.destage1x1 = tf.keras.layers.Conv2D(filters=12,kernel_size=1,strides=1,padding='same') 
    self.deconv2d_3 = tf.keras.layers.Conv2DTranspose(3,kernel_size=9, padding='same', strides=2)
    #self.Conv2DTS3x3 = tf.keras.layers.Conv2DTranspose(3,kernel_size=3, padding='same', strides=2)
    super(RepVGG, self).build(input_shape)
 
 
  def _make_stage(self, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    blocks = []
    for stride in strides:
      cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
      blocks.append(RepVGGBlock(planes, kernel_size=3,
                                strides=stride, padding='same', groups=cur_groups, deploy=self.deploy))
      self.in_planes = planes
      self.cur_layer_idx += 1
    return tf.keras.Sequential(blocks)
 
  def call(self,inputs):
    x = inputs
    x = self.conv0(x)
    x = self.stage1x1(x)
    x = self.stage1(x)
    x = self.destage1x1(x)
    x = self.deconv2d_3(x)
    return x
 
  def get_all_layers(self):
    layers = {}
    for layer in self.layers:
      if isinstance(layer, tf.keras.Sequential):
        for l in layer.layers:
          layers[layer.name+'/'+l.name]=l
      else:
        layers[layer.name]=layer
    return layers
 
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}
 
def create_RepVGG_A0(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 8, 1, 1], num_classes=num_classes,
                width_multiplier=[1, 1, 1, 1], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_A1(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_A2(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_B0(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_B1(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_B1g2(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)
 
def create_RepVGG_B1g4(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)
 
 
def create_RepVGG_B2(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_B2g2(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)
 
def create_RepVGG_B2g4(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)
 
 
def create_RepVGG_B3(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_B3g2(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[1, 1, 1, 1], override_groups_map=g2_map, deploy=deploy)
 
def create_RepVGG_B3g4(num_classes=1000, deploy=False):
  return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)
 
 
func_dict = {
  'RepVGG-A0': create_RepVGG_A0,
  'RepVGG-A1': create_RepVGG_A1,
  'RepVGG-A2': create_RepVGG_A2,
  'RepVGG-B0': create_RepVGG_B0,
  'RepVGG-B1': create_RepVGG_B1,
  'RepVGG-B1g2': create_RepVGG_B1g2,
  'RepVGG-B1g4': create_RepVGG_B1g4,
  'RepVGG-B2': create_RepVGG_B2,
  'RepVGG-B2g2': create_RepVGG_B2g2,
  'RepVGG-B2g4': create_RepVGG_B2g4,
  'RepVGG-B3': create_RepVGG_B3,
  'RepVGG-B3g2': create_RepVGG_B3g2,
  'RepVGG-B3g4': create_RepVGG_B3g4,
}
def get_RepVGG_func_by_name(name):
  return func_dict[name]
 

def repvgg_model_convert(model:RepVGG, model_name, build_shape,save_path=None):
  converted_weights = {}
  for name, layer in model.get_all_layers().items():
    if hasattr(layer, 'repvgg_convert'):
      kernel, bias = layer.repvgg_convert()
      converted_weights[name + '/conv2d/kernel:0'] = kernel.numpy()
      converted_weights[name + '/conv2d/bias:0'] = bias.numpy()
    elif isinstance(layer, tf.keras.layers.Conv2DTranspose):
      converted_weights[name + '/kernel:0'] = layer.kernel.numpy()
      converted_weights[name + '/bias:0'] = layer.bias.numpy()
    elif isinstance(layer, tf.keras.layers.Conv2D):
      converted_weights[name + '/kernel:0'] = layer.kernel.numpy()
      converted_weights[name + '/bias:0'] = layer.bias.numpy()        
    else:
      print(name, type(layer))
  build_func = get_RepVGG_func_by_name(model_name)
  deploy_model = build_func(deploy=True)
  deploy_model.build(build_shape)
  trainable_variables = deploy_model.trainable_variables
  for i, var in enumerate(converted_weights.values()):
    deploy_var = trainable_variables[i]
    deploy_var.assign(var)
    tf.print('deploy param: ', deploy_var.name, deploy_var.shape, tf.math.reduce_mean(deploy_var))
 
  if save_path is not None:
    deploy_model.save(save_path)
 
  return deploy_model
