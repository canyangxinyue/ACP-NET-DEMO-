

from models.backbone.ACP.CoAE import Acp_Net_model_name
import re

def AcpNet(model_name="CoAE_embed",**kwargs):
    if model_name=="CoAE_embed":
        return Acp_Net_model_name(model_name,**kwargs)
    

