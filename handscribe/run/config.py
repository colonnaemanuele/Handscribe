import yaml
import os
from handscribe.utils import get_parser

def flatten_dict(d, parent_key='', sep='.'):
    """Appiattisce dizionari annidati."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def is_nested_config_key(key, config_data):
    top_level_key = key.split('.')[0]
    return top_level_key in config_data and isinstance(config_data[top_level_key], dict)

def load_config():
    sparser = get_parser()
    p = sparser.parse_args()
    nested_configs = {}
    
    if p.config is not None:
        with open(p.config, "r") as f:
            try:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                config_data = yaml.load(f)

        flattened_config = flatten_dict(config_data)
        
        parser_keys = vars(p).keys()
        flat_configs = {}
        
        for k, v in flattened_config.items():
            if is_nested_config_key(k, config_data):
                top_level_key = k.split('.')[0]
                if top_level_key not in nested_configs:
                    nested_configs[top_level_key] = config_data[top_level_key]
            else:
                flat_configs[k] = v
                if k not in parser_keys:
                    print(f"Warning: Config key '{k}' not found in parser arguments")

        sparser.set_defaults(**flat_configs)

    args = sparser.parse_args()
    
    if p.config is not None:
        for config_name, config_value in nested_configs.items():
            setattr(args, config_name, config_value)

    if hasattr(args, 'dataset'):
        config_path = f"handscribe/configs/{args.dataset}.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
        else:
            print(f"Warning: Dataset config {config_path} not found.")

    return args