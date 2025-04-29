import os
import yaml
import re

def load_config(config_path):
    """Load config file with environment variable support and default values"""
    
    # Define a pattern to match environment variables ${ENV_VAR:-default} or $ENV_VAR
    env_pattern = re.compile(r'(?:\${([^}^{]+)})|(?:\$([a-zA-Z0-9_]+))')
    
    # Read the file and replace environment variables
    with open(config_path, 'r') as f:
        yaml_content = f.read()
        
        # Replace ${VAR:-default} or $VAR with the environment variable values
        def replace_env_var(match):
            matched_var = match.group(1) or match.group(2)
            
            # Check if there's a default value specified with :- syntax
            if ':-' in matched_var:
                env_var, default = matched_var.split(':-', 1)
                return os.environ.get(env_var, default)
            else:
                return os.environ.get(matched_var, '')
        
        processed_yaml = env_pattern.sub(replace_env_var, yaml_content)
        
        # Load the processed YAML
        return yaml.safe_load(processed_yaml)