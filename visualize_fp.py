import matplotlib.pyplot as plt
import mmcv
from flashtorch.saliency import Backprop
from flashtorch.utils import apply_transforms
from flashtorch.activmax import GradientAscent
import torch

from mmseg.apis import init_model

def get_conv_weights(model, layer_name):
    """Get the convolutional layer weights from a model given the layer name."""
    layer = dict(model.named_modules()).get(layer_name, None)
    if layer is None:
        raise ValueError(f'Layer {layer_name} not found in the model.')
    if not isinstance(layer, torch.nn.Conv2d):
        raise ValueError(f'Layer {layer_name} is not a Conv2d layer.')
    return layer.weight.data.cpu().numpy()

# 指定配置文件和检查点文件的路径
config_file = 'work_dirs/unet_baseline/unet_baseline.py'
checkpoint_path = 'work_dirs/unet_baseline/iter_4000.pth'

# Init the model from the config and the checkpoint
model = init_model(config_file, checkpoint_path, 'cpu')
#
# img_path = '/Users/jessica/PycharmProjects/mmsegmentation/data/DRIVE/images/training/21.png'
#
# # 读取图片
# img = mmcv.imread(img_path)
#
# # 将图像转换为模型输入格式
# img = apply_transforms(img)
#
# # 将图像移动到相同的设备（'mps'）
# img = img.to('mps')

# 获取模型中的某一层（例如，'backbone.encoder[3][1].convs[0]）
target_layer = eval('model.backbone.encoder[4][1].convs[1].conv')

# 创建GradientAscent对象,
# 要传入整个model
g_ascent = GradientAscent(model)
# 指定要可视化的filter的index
filters = [0, 511,1023]
# 获取模型中的某一层（例如，'backbone.encoder[3][1].convs[0]）
target_layer = eval('model.backbone.encoder[4][1].convs[1].conv')
# 确保选择的层是 nn.Conv2d 类型
if not isinstance(target_layer, torch.nn.Conv2d):
    raise TypeError(f'The layer must be nn.Conv2d.')
# 可视化卷积核
g_ascent.visualize(target_layer, filters)
plt.show()
