
def Acp_Net_model_name(model_name="ACPNet",**kwargs):



    if model_name=="CoAE_embed":
        from .ACP import ACPNet
        return ACPNet(**kwargs)

