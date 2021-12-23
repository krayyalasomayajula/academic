import os

def get_class_from_str(class_str):
    class_strings = class_str.split('.')
    config_mod, config_cls = '.'.join(class_strings[:-1]), class_strings[-1]
    #print(config_mod, config_cls)
    mod = __import__(config_mod, fromlist=[config_cls]) #from my_package.my_module import my_class
    klass = getattr(mod, config_cls) #get class instance
    #print(klass)
    return klass

def get_class_from_key(config):
    klass = get_class_from_str(list(config.keys())[0])
    return klass

def get_filenames(path):
    """Returns a list of absolute paths to images inside given 'path' """
    files_list = []
    for filename in os.listdir(path):
        files_list.append(os.path.join(path, filename))
    return files_list
    