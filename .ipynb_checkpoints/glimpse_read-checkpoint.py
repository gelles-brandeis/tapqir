import numpy as np
import torch

def load_aois(header, aoi_list, loc, drift, frames):
    smd = {}
    smd["indices"] = aoi_list
    smd["frames"] = frames
    smd["pixnum"] = int(loc["aoiinfo2"][0,4])
    smd["height"] = int(header["vid"]["height"])
    smd["width"] = int(header["vid"]["width"])
    smd["traces"] = []
    smd["data"] = np.zeros((len(smd["indices"]),len(smd["frames"]),smd["pixnum"],smd["pixnum"]))
    
    for i in smd["indices"]:
        trace = {}
        trace["id"] = i
        #trace["frames"] = frames
        idx = loc["aoiinfo2"][:,5] == i
        trace["xy_loc"] = loc["aoiinfo2"][idx,2:4] + drift["driftlist"][frames,1:3]
        if smd["pixnum"] % 2:
            trace["xy_pixel"] = np.round(trace["xy_loc"])
        else:
            trace["xy_pixel"] = np.floor(trace["xy_loc"]) + 0.5
        
        #trace["data"] = np.zeros((len(smd["frames"]),smd["pixnum"],smd["pixnum"]))
        smd["traces"].append(trace)
        
    for j, f in enumerate(smd["frames"]):
        glimpse_number = header["vid"]["filenumber"][0][0][0][f]
        with open("/home/ordabayev/Documents/postdoc/Bayesian_test_files/b33p43e_440/{}.glimpse".format(glimpse_number)) as fid:
            fid.seek(header['vid']['offset'][0][0][0][f])
            img = np.fromfile(fid, dtype='>i2', count=smd["height"]*smd["width"]).reshape(smd["height"],smd["width"])
            img += 2**15
            img.astype(np.uint16)
            
            for k in range(len(smd["indices"])):
                x1 = int(smd["traces"][k]["xy_pixel"][j,0] - (smd["pixnum"] - 1)/2 - 1)
                x2 = int(smd["traces"][k]["xy_pixel"][j,0] + (smd["pixnum"] - 1)/2)
                y1 = int(smd["traces"][k]["xy_pixel"][j,1] - (smd["pixnum"] - 1)/2 - 1)
                y2 = int(smd["traces"][k]["xy_pixel"][j,1] + (smd["pixnum"] - 1)/2)
                smd["data"][k,j,:,:] = img[x1:x2,y1:y2]
                
    smd["data"] = torch.tensor(smd["data"], dtype=torch.float32)
            
    return smd