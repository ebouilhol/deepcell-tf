from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deepcell.homeapplications import homeMadeModel

from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import tf_export



@tf_export('homeMadeModel.ResNet50_3D',
           'homeMadeModel.ResNet50_3D')
@keras_modules_injection
def ResNet50_3D(*args, **kwargs):
  return homeMadeModel.ResNet50_3D(*args, **kwargs)
