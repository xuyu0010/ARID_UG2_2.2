import logging

def get_config(name, **kwargs):

    logging.debug("loading network configs of: {}".format(name.upper()))

    config = {}

    logging.info("Preprocessing:: using Video default mean & std.")
    config['mean'] = [124 / 255, 117 / 255, 104 / 255]
    config['std'] = [1 / (.0167 * 255)] * 3
    
    # config['mean'] = [0.485, 0.456, 0.406]    # ImageNet config
    # config['std'] = [0.229, 0.224, 0.225]    # ImageNet config

    logging.info("data:: {}".format(config))
    return config