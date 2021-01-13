import logging

from .r3d_18_da import r3d_18_da
from .config import get_config

def get_symbol(name, print_net=False, DA_method=None, **kwargs):

	logging.info("Network:: Getting symbol with {} domain adaptation using {} network.".format(DA_method, name))

	if "R3D18" in name.upper():
		net = r3d_18_da(DA_method=DA_method, **kwargs)
	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, **kwargs)
	return net, input_conf
