// Example folder structure
/*
{
  "display_name": "All Files",
  "full_path": "/",
  "type": "root",
  "subfolders": [
    {
      "display_name": "Local",
      "full_path": "osfs:///mnt/media/",
      "base_url": "osfs://",
      "path_in_fs": "/mnt/media/",
      "type": "server",
      "num_files": 10,
      "total_files": 30,
      "subfolders": [
        {
          "display_name": "Folder 1",
          "full_path": "osfs:///mnt/media/Folder 1/",
          "base_url": "osfs://",
          "path_in_fs": "/mnt/media/Folder 1/",
          "type": "folder",
          "num_files": 5,
          "total_files": 15,
          "subfolders": []
        },
        {
          "display_name": "Folder 2",
          "full_path": "osfs:///mnt/media/Folder 2/",
          "base_url": "osfs://",
          "path_in_fs": "/mnt/media/Folder 2/",
          "type": "folder",
          "num_files": 5,
          "total_files": 15,
          "subfolders": [
             // Deeper nested folders...
          ]
        }
      ]
    },
    {
      "display_name": "My Home Server (192.168.0.19:5002)",
      "full_path": "webdav://192.168.0.19:5002/",
      "base_url": "webdav://192.168.0.19:5002/",
      "path_in_fs": "/",
      "type": "server",
      "subfolders": []
    },
    {
      "display_name": "Friend's Images (frends.server.fr)",
      "full_path": "ftp://frends.server.fr/images/",
      "base_url": "ftp://frends.server.fr/",
      "path_in_fs": "/images/",
      "type": "server",
      "subfolders": []
    }
  ]
}
*/

/**
 * Class representing a folder view component.
 */
class FolderViewComponent {
  /**
   * Create a FolderViewComponent.
   * @param {Object} folders_dict - The dictionary representing the folder structure.
   * @param {string} [active_path=''] - The active path to highlight.
   * @param {boolean} [enableContextMenu=false] - Whether to enable the right-click context menu.
   */
  constructor(folders_dict, active_path = '', enableContextMenu = false) {
    this.folders_dict = folders_dict;
    this.active_path = active_path;
    this.enableContextMenu = enableContextMenu;
    this.contextMenu = null;
    this.currentContextPath = '';

    // Only initialize context menu elements if enabled
    if (this.enableContextMenu) {
      this.contextMenu = document.getElementById('folder-context-menu');
      if (!this.contextMenu) {
        console.warn('Context menu element not found. Make sure #folder-context-menu exists in the DOM.');
      }
    }

    // Create the DOM element
    this.dom_element = document.createElement('ul');
    this.dom_element.className = 'menu-list';
    const li = this.createFolderRepresentation(this.folders_dict, this.active_path);
    this.dom_element.appendChild(li);

    // Set up context menu event listeners only if enabled
    if (this.enableContextMenu && this.contextMenu) {
      this.setupContextMenu();
    }
  }

  /**
   * Check if a path is a subfolder of another path.
   * @param {string} parentPath - The parent path.
   * @param {string} childPath - The child path.
   * @returns {boolean} True if the child path is a subfolder of the parent path, false otherwise.
   */
  isSubfolder(parentPath, childPath) {
    // Normalize paths to remove any '..' or '.' segments
    const normalizedParentPath = new URL(parentPath, 'file://').pathname.replace(/\/$/, '') + '/';
    const normalizedChildPath = new URL(childPath, 'file://').pathname.replace(/\/$/, '') + '/';

    // Check if the child path starts with the parent path
    return normalizedChildPath.startsWith(normalizedParentPath);
  }

  /**
   * Create a folder representation as a DOM element.
   * @param {Object} folders_dict - The dictionary representing the folder structure.
   * @param {string} [active_path=''] - The active path to highlight.
   * @param {string} [current_path=''] - The current path being processed.
   * @returns {HTMLElement} The DOM element representing the folder.
   */
  createFolderRepresentation(folder_node, active_path = '') {
    const folderRepresentation = document.createElement('li');

    // 1. Read explicitly from the new JSON structure
    const displayName = folder_node.display_name || folder_node.name;
    const currentPath = folder_node.full_path; // No more string concatenation!
    const numFiles = folder_node.num_files;
    const totalFiles = folder_node.total_files;

    // 2. Determine UI active state and colors
    const isActive = active_path === currentPath ? 'is-active' : '';
    let color = this.isSubfolder(active_path, currentPath) ? 'has-text-strong' : 'has-text-grey-light';
    if (active_path === currentPath) color = '';

    // 3. Optional: File counts (hide them if they aren't provided for remote servers)
    let imageCountDisplay = '';
    if (numFiles !== undefined && totalFiles !== undefined) {
      imageCountDisplay = numFiles === totalFiles ? `[${numFiles}]` : `[${numFiles} | ${totalFiles}]`;
    }

    // 4. Create the anchor tag
    const a = document.createElement('a');
    a.className = `${isActive} ${color}`;
    a.href = `?path=${encodeURIComponent(currentPath)}`;

    if (folder_node.type === 'server') {
        const dot = document.createElement('span');
        dot.className = 'server-dot';
        if (folder_node.is_available === true) {
            dot.classList.add('is-available');
        } else if (folder_node.is_available === false) {
            dot.classList.add('is-unavailable');
        } else {
            dot.classList.add('is-pending');
        }
        a.appendChild(dot);
    }

    a.appendChild(document.createTextNode(`${displayName} ${imageCountDisplay}`.trim()));

    // Context Menu logic (unchanged)
    if (this.enableContextMenu && this.contextMenu) {
      a.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        this.showContextMenu(e, currentPath);
      });
    }

    folderRepresentation.appendChild(a);

    // 5. Recursively render subfolders
    // Now we can just iterate over the array securely instead of Object.keys()
    if (folder_node.subfolders && folder_node.subfolders.length > 0) {
      const ul = document.createElement('ul');

      for (const childFolder of folder_node.subfolders) {
        // Only render children if they are in the active path expansion
        // if (this.isSubfolder(childFolder.full_path, active_path)) {
          ul.appendChild(this.createFolderRepresentation(childFolder, active_path));
        //}
      }
      folderRepresentation.appendChild(ul);
    }

    return folderRepresentation;
  }

  /**
   * Get the DOM element representing the folder view.
   * @returns {HTMLElement} The DOM element representing the folder view.
   */
  getDOMElement() {
    return this.dom_element;
  }

  /**
   * Setup the context menu for folder actions.
   */
  setupContextMenu() {
    // Hide context menu when clicking elsewhere
    document.addEventListener('click', (e) => {
      if (!this.contextMenu.contains(e.target)) {
        this.hideContextMenu();
      }
    });

    // Context menu actions
    document.getElementById('create-folder').addEventListener('click', (e) => {
      e.preventDefault();
      this.createFolder();
    });

    document.getElementById('create-file').addEventListener('click', (e) => {
      e.preventDefault();
      this.createFile();
    });

    document.getElementById('rename-folder').addEventListener('click', (e) => {
      e.preventDefault();
      this.renameFolder();
    });

    document.getElementById('delete-folder').addEventListener('click', (e) => {
      e.preventDefault();
      this.deleteFolder();
    });
  }

  /**
   * Show the context menu for folder actions.
   * @param {MouseEvent} event - The mouse event triggering the context menu.
   * @param {string} folderPath - The path of the folder for context actions.
   */
  showContextMenu(event, folderPath) {
    this.currentContextPath = folderPath;
    
    // Position the context menu
    this.contextMenu.style.left = `${event.pageX}px`;
    this.contextMenu.style.top = `${event.pageY}px`;
    
    // Show the context menu
    this.contextMenu.classList.remove('is-hidden');
  }

  /**
   * Hide the context menu.
   */
  hideContextMenu() {
    this.contextMenu.classList.add('is-hidden');
  }


  /**
   * Create a new folder.
   */
  createFolder() {
    const folderName = prompt('Enter folder name:');
    if (folderName && folderName.trim()) {
      // Emit socket event to create folder
      socket.emit('emit_create_folder', {
        path: this.currentContextPath,
        name: folderName.trim()
      });
    }
    this.hideContextMenu();
  }

  /**
   * Create a new file.
   */
  createFile() {
    const fileName = prompt('Enter file name:');
    if (fileName && fileName.trim()) {
      // Emit socket event to create file
      socket.emit('emit_create_file', {
        path: this.currentContextPath,
        name: fileName.trim()
      });
    }
    this.hideContextMenu();
  }

  /**
   * Rename a folder.
   */
  renameFolder() {
    const currentName = this.currentContextPath.split('/').slice(-2)[0]; // Get folder name
    const newName = prompt('Enter new folder name:', currentName);
    if (newName && newName.trim() && newName !== currentName) {
      socket.emit('emit_rename_folder', {
        path: this.currentContextPath,
        newName: newName.trim()
      });
    }
    this.hideContextMenu();
  }

  /**
   * Delete a folder.
   */
  deleteFolder() {
    if (confirm(`Are you sure you want to delete this folder: ${this.currentContextPath}?`)) {
      socket.emit('emit_delete_folder', {
        path: this.currentContextPath
      });
    }
    this.hideContextMenu();
  }
}

export default FolderViewComponent;