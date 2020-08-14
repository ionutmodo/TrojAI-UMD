"""This file contains global constants and methods to be used anywhere in the project"""
import socket

TrojAI_input_size = (1, 3, 224, 224)
TrojAI_num_classes = 5

def get_project_root_path():
    """
    Returns the root path of the project on a specific machine.
    To add your custom path, you need to add an entry in the dictionary.
    :return:
    """
    hostname = socket.gethostname()
    hostname = 'openlab' if hostname.startswith('openlab') else hostname
    hostname_root_dict = { # key = hostname, value = your local root path
        'ubuntu20': '/mnt/storage/Cloud/MEGA/TrojAI',  # the name of ionut's machine
        'openlab': '/fs/sdsatumd/ionmodo/TrojAI' # name of UMD machine
    }
    print(f'Running on machine "{hostname}"')

    return hostname_root_dict[hostname]