import basicsr as rt
from ..utils.lib import *
from basicsr.models.archs.restormer_arch import Restormer
yaml_file="/Users/azuma/PycharmProjects/bmecompetition/Restormer/Motion_Deblurring/Options/Deblurring_Restormer.yml"
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
pthpath= "/Users/azuma/PycharmProjects/bmecompetition/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth"
def getmodel(pthpath=pthpath, yaml_file=yaml_file):
    """
    :param pthpath: path to the model checkpoint
    :param yaml_file: path to the yaml file
    :return: model,inputsize
    """
    restransckpt = torch.load(pthpath)
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
    _ = x['network_g'].pop('type')
    model_restoration = Restormer(**x['network_g'])
    model_restoration.load_state_dict(restransckpt['params'])
    return model_restoration,(192,128)