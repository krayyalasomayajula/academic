from copy import deepcopy

def parse_loader_config(loader_yaml_config):
    assert isinstance(loader_yaml_config, dict), 'Error expecting a config of type dict'
    assert all(x in loader_yaml_config.keys() for x in ['general','train','valid'])




