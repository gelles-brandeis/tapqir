import pyro
# pyro & pytorch
import torch
import argparse

device = torch.device("cuda:0")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.is_available()

from Modelv1 import Modelv1p1
#from Modelv2 import Modelv2

from classifier import select_model, run_model

parser = argparse.ArgumentParser(description="classifier")
#parser.add_argument("-m", "--model", default="1", type=str,
                        #help="one of: {}".format(", ".join(sorted(models.keys()))))
parser.add_argument("-n", "--num-epochs", default=50, type=int)
parser.add_argument("-N", "--n-batch", default=16, type=int)
parser.add_argument("-F", "--f-batch", default=64, type=int)
args = parser.parse_args()

pyro.clear_param_store()

svi, model, writer = select_model("Modelv1p1", Modelv1p1, args.n_batch, args.f_batch)

epoch_count = 0

run_model("Modelv1p1", svi, model, writer, epoch_count, n_epochs=args.num_epochs, n_batch=args.n_batch, f_batch=args.f_batch)
