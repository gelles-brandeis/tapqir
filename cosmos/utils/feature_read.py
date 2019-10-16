import os
# pyro & pytorch
import torch
# loading & saving data
import pyro

from cosmos.utils.aoi_reader import ReadAoi 

def ReadFeatures(data, dataset, device):
    #data = ReadAoi(dataset, device)

    #attributes = ["h_loc", "h_beta", "b_loc", "b_beta", "w_loc", "w_beta", "x_loc", "x_scale", "y_loc", "y_scale"]
    #for attr in attributes:
    #    setattr(data, attr, torch.load(os.path.join(data.path, "runs", "features", "{}.pt".format(attr)), map_location=device))
    #pyro.get_param_store().load(os.path.join(data.path, "runs", dataset, "features", "params"))
    #for attr in pyro.get_param_store().keys():
    #    setattr(data, attr, pyro.param(attr).detach().squeeze().to(device))

    pyro.get_param_store().load(os.path.join(data.path, "runs", dataset, "junk", "K1", "params"))
    for attr in pyro.get_param_store().keys():
        setattr(data, attr, pyro.param(attr).detach().squeeze().to(device))

    pyro.get_param_store().load(os.path.join(data.path, "runs", dataset, "junk", "K2", "params"))
    for attr in pyro.get_param_store().keys():
        setattr(data, attr, pyro.param(attr).detach().squeeze().to(device))

    print("Features were read from saved files.")

    return data
