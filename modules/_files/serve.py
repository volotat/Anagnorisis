from src.socket_events import CommonSocketEvents
# from .vfs import VFSManager

class FilesModuleServer:
    """Controller for the Native Files extension."""
    
    def __init__(self, app, socketio, cfg, data_folder):
        self.app = app
        self.socketio = socketio
        self.cfg = cfg
        self.events = CommonSocketEvents(socketio, module_name="files")
        # self.vfs = VFSManager()

    def initialize(self):
        self.events.show_loading_status('Initializing Virtual File System...')
        # self.vfs.load_user_servers()
        self._register_socket_events()
        self.events.show_loading_status('Files module ready!')

    def _register_socket_events(self):
        self.socketio.on_event('emit_files_page_get_folders', self.handle_get_folders)
        self.socketio.on_event('emit_files_page_get_files', self.handle_get_files)

    # -------------------------------------------------------------------------
    # SOCKET EVENT HANDLERS
    # -------------------------------------------------------------------------

    def handle_get_folders(self, data):
        """Builds and returns the nested folder tree based on the active path."""
        active_path = data.get('path', '')
        folder_tree = self._build_folder_tree(active_path)
        return folder_tree # Or {"folders": folder_tree} if frontend expects it

    def handle_get_files(self, data):
        virtual_path = data.get('path', '')
        files = self.vfs.get_files(virtual_path)
        return {"files_data": files}

    # -------------------------------------------------------------------------
    # TREE BUILDER LOGIC
    # -------------------------------------------------------------------------

    def _build_folder_tree(self, active_path: str) -> dict:
        """
        Dynamically builds the nested JSON array expected by FolderViewComponent.js.
        It expands the tree ONLY along the active_path to prevent network lag.
        """
        # 1. Initialize the Root Node
        root_node = {
            "display_name": "All Files",
            "name": "root",
            "path": "/",
            "type": "root",
            "subfolders": []
        }

        # 2. Add all servers to the root
        servers = self.vfs.get_folders("/")
        server_nodes = {}
        for srv in servers:
            node = {**srv, "subfolders": []}
            root_node["subfolders"].append(node)
            server_nodes[srv["name"]] = node  # Keep a fast reference

        clean_path = active_path.strip('/')
        if not clean_path or ':' not in clean_path:
            return root_node

        source, rest = clean_path.split(':', 1)
        if source not in server_nodes:
            return root_node

        current_node = server_nodes[source]
        current_vfs_path = f"{source}:/"

        # 3. Expand the base of the selected server
        try:
            children = self.vfs.get_folders(current_vfs_path)
            for child in children:
                current_node["subfolders"].append({**child, "subfolders": []})
        except Exception:
            return root_node

        # 4. Traverse down the active path segments
        segments = [s for s in rest.split('/') if s]
        for segment in segments:
            # Find the next node in the subfolders array
            next_node = next((n for n in current_node["subfolders"] if n["name"] == segment), None)
            if not next_node:
                break
            
            current_node = next_node
            current_vfs_path += f"{segment}/"

            # Fetch children for this newly opened level
            try:
                children = self.vfs.get_folders(current_vfs_path)
                for child in children:
                    current_node["subfolders"].append({**child, "subfolders": []})
            except Exception:
                break

        return root_node

def register_module(app, socketio, cfg, data_folder):
    module_server = FilesModuleServer(app, socketio, cfg, data_folder)
    module_server.initialize()
    return module_server