'''
This repository serves as the base structure for UG2+ 2022 Challenge Track 2: Semi-supervised AR in the Dark.
This repository is based on the repository at https://github.com/cypw/PyTorch-MFNet. We thank the authors for the repository.
This repository is authored by Yuecong Xu, please contact at xuyu0014 at e.ntu.edu.sg

Note: this repository could only be used when CUDA is available!!!
'''
import os
import json
import socket
import logging
import argparse
from datetime import date

import torch
import torch.nn.parallel
import torch.distributed as dist

import dataset
from train_da_model import train_da_model
from network.symbol_builder import get_symbol

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
# debug
parser.add_argument('--debug-mode', type=bool, default=True, help="print all setting for debugging.")

# io
parser.add_argument('--dataset', default='UG2-2022')
parser.add_argument('--source-dataset', default='CLEAR')
parser.add_argument('--target-dataset', default='ARID')
parser.add_argument('--clip-length', default=16, help="define the length of each input sample.")
parser.add_argument('--train-frame-interval', type=int, default=2,help="define the sampling interval between frames.")

parser.add_argument('--task-name', type=str, default='',help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./exps/models",help="set logging file.")
parser.add_argument('--log-file', type=str, default="",help="set logging file.")

# device
parser.add_argument('--gpus', type=str, default="0",help="define gpu id")

# algorithm
parser.add_argument('--network', type=str, default='R3D18',help="choose the base network")
parser.add_argument('--DA-method', default=None, help="applied domain adaptation method (e.g. DANN, MCD, MMD), DANN loss (adversarial domain loss) is provided for reference")
parser.add_argument('--dom-rev', type=float, default=1.0, help="available when DA-method is DANN: domain reverse weight (gradient reverse layer weight)")
parser.add_argument('--resume-epoch', type=int, default=-1, help="resume train")

# optimization
parser.add_argument('--fine-tune', type=bool, default=True, help="apply different learning rate for different layers")
parser.add_argument('--batch-size', type=int, default=4, help="batch size")
parser.add_argument('--lr-base', type=float, default=0.01, help="learning rate")
parser.add_argument('--lr-steps', type=list, default=[int(1e4*x) for x in [1, 2, 4]], help="# of samples before changing lr")
parser.add_argument('--lr-factor', type=float, default=0.1, help="reduce the lr with factor")

# other training parameters
parser.add_argument('--save-frequency', type=float, default=1, help="save once after N epochs")
parser.add_argument('--end-epoch', type=int, default=2, help="maxmium number of training epoch")
parser.add_argument('--random-seed', type=int, default=1, help='random seed (default: 1)')

def autofill(args):
	# customized
	if not args.task_name:
		args.task_name = os.path.basename(os.getcwd())
	if not args.log_file:
		if os.path.exists("./exps/logs"):
			args.log_file = "./exps/logs/{}_at-{}.log".format(args.task_name, socket.gethostname())
		else:
			args.log_file = ".{}_at-{}.log".format(args.task_name, socket.gethostname())
	# fixed
	args.model_prefix = os.path.join(args.model_dir, args.task_name)
	return args

def set_logger(log_file='', debug_mode=False):
	if log_file:
		if not os.path.exists("./"+os.path.dirname(log_file)):
			os.makedirs("./"+os.path.dirname(log_file))
		handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
	else:
		handlers = [logging.StreamHandler()]

	""" add '%(filename)s:%(lineno)d %(levelname)s:' to format show source file """
	logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers = handlers)

if __name__ == "__main__":

	# set args
	args = parser.parse_args()
	args = autofill(args)

	set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
	logging.info("Using pytorch {} ({})".format(torch.__version__, torch.__path__))
	logging.info("Start training with args:\n" + json.dumps(vars(args), indent=4, sort_keys=True))

	# set device states
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
	assert torch.cuda.is_available(), "CUDA is not available"
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)

	# load dataset related configuration
	dataset_cfg = dataset.get_config(name=args.dataset)

	# creat model with all parameters initialized
	net, input_conf = get_symbol(name=args.network, pretrained=True, DA_method=args.DA_method, **dataset_cfg)

	# training
	kwargs = {}
	kwargs.update(dataset_cfg)
	kwargs.update({'input_conf': input_conf})
	kwargs.update(vars(args))
	train_da_model(sym_net=net, **kwargs)
