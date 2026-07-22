import torch

#torch.no_grad disables the autograd engine, therefore no computation graph will be made, thus reducing the unnecessary compute

"""
The decay controls the half-life of the exponential window.
Eg: for d=0.995 half life=138 steps. means 138 steps ago contributes 50% of the current weights, which is good for small amount of training
but may need to increase for larger training times
"""
@torch.no_grad()
def update_ema(ema_model, model, decay=0.995):
  ema_params=dict(ema_model.named_parameters())
  model_params=dict(model.named_parameters())

  for k in ema_params.keys():
    ema_params[k].data.mul_(decay).add_(model_params[k].data, alpha=1-decay)

#.mul and .add do inplace operations to reduce memory requirements, but we do not do it when using autograd as it destorys the original the
#original variable so during backpropagation when model looks for the original value of the variable this creates a problem