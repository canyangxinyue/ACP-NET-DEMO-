from .abstract_weighting import AbsWeighting
from .EW import EW
from .GradNorm import GradNorm
from .MGDA import MGDA
from .UW import UW 
from .DWA import DWA
from .GLS import GLS
from .GradDrop import GradDrop
from .PCGrad import PCGrad
from .GradVac import GradVac
from .IMTL import IMTL
from .CAGrad import CAGrad
from .Nash_MTL import Nash_MTL
from .RLW import RLW
from .MoCo import MoCo
from .Aligned_MTL import Aligned_MTL
from .DB_MTL import DB_MTL

__all__ = ['AbsWeighting',
           'EW', 
           'GradNorm', 
           'MGDA',
           'UW',
           'DWA',
           'GLS',
           'GradDrop',
           'PCGrad',
           'GradVac',
           'IMTL',
           'CAGrad',
           'Nash_MTL',
           'RLW',
           'MoCo',
           'Aligned_MTL',
           'DB_MTL',
           'build_weighting']

# 需要自己设置self.task_num

def build_weighting(name, **config):
    assert name in __all__, f'all support loss is {__all__}'
    weighting_method = eval(name)()
    weighting_method.kwargs=config
    return weighting_method