import os
import glob
from omegaconf import OmegaConf

from anagnorisis_core.media.media_types import bundled_media_types_dir, get_registry

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

        # Auto-merge the media-type taxonomy (media_types/*.yaml), same override
        # rules as module defaults. Tag vocabularies in media_types/tags/ are
        # deliberately NOT merged here — MediaTypeRegistry loads them, so the
        # bulky lists never reach the config or the embedding subprocesses.
        # The taxonomy ships inside anagnorisis_core. A `media_types/` folder at
        # the repository root, if present, overrides it — that is where a user
        # puts their own extension lists or tag vocabularies without editing the
        # package.
        media_types_folder = bundled_media_types_dir()
        override_folder = os.path.join(root_folder, 'media_types')
        if os.path.isdir(override_folder):
            media_types_folder = override_folder
        for types_cfg_path in sorted(glob.glob(os.path.join(media_types_folder, '*.yaml'))):
            cfg = OmegaConf.merge(OmegaConf.load(types_cfg_path), cfg)
            print(f"Merged media type definitions: {types_cfg_path}")

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
            "memory": os.path.join(project_config_folder_path, 'memory'),
        }

        # Create necessary directories
        for path in [
            project_config_folder_path,
            os.path.dirname(paths["database"]),
            paths["personal_models"],
            paths["cache"],
            paths["memory"]
        ]:
            os.makedirs(path, exist_ok=True)

        # Inject final paths into OmegaConf
        cfg.main.database_path = paths["database"]
        cfg.main.migrations_path = paths["migrations"]
        cfg.main.embedding_models_path = paths["embedding_models"]
        cfg.main.personal_models_path = paths["personal_models"]
        cfg.main.cache_path = paths["cache"]
        cfg.main.memory_path = paths["memory"]
        cfg.main.media_types_path = media_types_folder

        # Derive each module's media_formats from the media types it declares, so
        # extension lists exist in exactly one place. A module that declares
        # `media_types:` must not also set `media_formats:` — the derived value
        # wins. Modules with formats of their own (e.g. YouTube's .link) simply
        # declare no media types and keep their explicit list.
        registry = get_registry(cfg)
        for section in cfg:
            declared = OmegaConf.select(cfg, f'{section}.media_types', default=None)
            if declared:
                cfg[section].media_formats = registry.extensions_for(declared)

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