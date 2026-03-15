// Example folder structure
const folders_dict = {
  name: 'root',
  num_files: 10,
  total_files: 30,
  subfolders: {
    folder1: {
      name: 'folder1',
      num_files: 5,
      total_files: 15,
      subfolders: {
        subfolder1: {
          name: 'subfolder1',
          num_files: 3,
          total_files: 3,
          subfolders: {}
        }
      }
    },
    folder2: {
      name: 'folder2',
      num_files: 5,
      total_files: 15,
      subfolders: {}
    }
  }
};

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
  createFolderRepresentation(folders_dict, active_path = '', current_path = '') {
    const folderRepresentation = document.createElement('li');

    const folderName = folders_dict.name;
    const numFiles = folders_dict.num_files;
    const totalFiles = folders_dict.total_files;
    let current_path_ = current_path + folderName + '/';
    const isActive = active_path === current_path_ ? 'is-active' : '';
    let encoded_link = `path=${encodeURIComponent(current_path_)}`;
    let color = this.isSubfolder(active_path, current_path_) ? 'has-text-black' : 'has-text-grey-light';
    if (active_path === current_path_) color = '';

    // Conditional check for displaying image counts
    let imageCountDisplay = numFiles === totalFiles ? `[${numFiles}]` : `[${numFiles} | ${totalFiles}]`;

    const a = document.createElement('a');
    a.className = `${isActive} ${color}`;
    a.href = `?${encoded_link}`;
    a.textContent = `${folderName} ${imageCountDisplay}`;

    // Add right-click context menu only if enabled
    if (this.enableContextMenu && this.contextMenu) {
      a.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        this.showContextMenu(e, current_path_);
      });
    }

    folderRepresentation.appendChild(a);

    // Sort the folders by name
    const sortedFolders = Object.keys(folders_dict.subfolders).sort((a, b) => {
      return folders_dict.subfolders[a].name.localeCompare(folders_dict.subfolders[b].name);
    });

    const ul = document.createElement('ul');

    // Create a new folder representation for each subfolder
    for (const folderKey of sortedFolders) {
      const folder = folders_dict.subfolders[folderKey];
      let current_path_ = current_path + folderName + '/';

      if (this.isSubfolder(current_path_, active_path)) {
        ul.appendChild(this.createFolderRepresentation(folder, active_path, current_path_));
      }
    }

    folderRepresentation.appendChild(ul);
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