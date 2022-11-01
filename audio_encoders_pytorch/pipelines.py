# from typing import Dict, Optional, Sequence, Tuple, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor

# from .modules import MAE, Discriminator1d
# from .utils import prefix_dict, to_list


# def freeze_model(model: nn.Module):
#     for param in model.parameters():
#         param.requires_grad = False


# def unfreeze_model(model: nn.Module):
#     for param in model.parameters():
#         param.requires_grad = True
