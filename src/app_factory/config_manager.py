import os
import glob
from omegaconf import OmegaConf

class ConfigManager:
    """Handles parsing configuration files and ensuring required directories exist."""

    @classmethod
    def setup(cls, root_folder):
        print(f"Script folder: {root_folder}")
        
        # Create logs directory if it doesn't exist
        logs_folder = os.path.join(root_folder, 'logs')
        os.makedirs(logs_folder, exist_ok=True)
        print(f"Using data folder: {root_folder}")

        # Load default configuration
        config_path = os.path.join(root_folder, 'config.yaml')
        cfg = OmegaConf.load(config_path)

        # Auto-merge module config defaults (modules/<module>/config.defaults.yaml).
        # Module defaults are loaded first, then the root config is applied on top,
        # so root config.yaml always wins as an override.
        modules_pattern = os.path.join(root_folder, 'modules', '*', 'config.defaults.yaml')
        for mod_cfg_path in sorted(glob.glob(modules_pattern)):
            mod_cfg = OmegaConf.load(mod_cfg_path)
            cfg = OmegaConf.merge(mod_cfg, cfg)
            print(f"Merged module config defaults: {mod_cfg_path}")

        # Load local configuration if it exists
        project_config_folder_path = cfg.main.get('project_config_directory', 'project_config')
        if not os.path.isabs(project_config_folder_path):
            project_config_folder_path = os.path.join(root_folder, project_config_folder_path)

        paths = {
            "database": os.path.join(project_config_folder_path, 'database', 'project.db'),
            "migrations": os.path.join(project_config_folder_path, 'database', 'migrations'),
            "embedding_models": os.path.join(root_folder, 'models'),
            "personal_models": os.path.join(project_config_folder_path, 'models'),
            "cache": os.path.join(project_config_folder_path, 'cache'),
        }

        # Create necessary directories
        for path in [
            project_config_folder_path,
            os.path.dirname(paths["database"]),
            paths["personal_models"],
            paths["cache"]
        ]:
            os.makedirs(path, exist_ok=True)

        # Inject final paths into OmegaConf
        cfg.main.database_path = paths["database"]
        cfg.main.migrations_path = paths["migrations"]
        cfg.main.embedding_models_path = paths["embedding_models"]
        cfg.main.personal_models_path = paths["personal_models"]
        cfg.main.cache_path = paths["cache"]

        # Load the user-specific configuration as a clean, isolated object
        user_config_path = os.path.join(project_config_folder_path, 'config.yaml')
        user_cfg = OmegaConf.create()  # Default to empty config if not present on disk
        user_cfg.servers = [{
            "name": "Local",
            "url": "osfs:///mnt/media/"
        }]
        if os.path.exists(user_config_path):
            user_cfg = OmegaConf.load(user_config_path)
            print(f"Loaded local user config: {user_config_path}")
        else:
            OmegaConf.save(user_cfg, user_config_path)
            print(f"Created default user config: {user_config_path}")

        print(f"Database path set to: {paths['database']}")
        return cfg, user_cfg, paths