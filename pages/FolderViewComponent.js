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
   */
  constructor(folders_dict, active_path = '') {
    this.folders_dict = folders_dict;
    this.active_path = active_path;

    // Create the DOM element
    this.dom_element = document.createElement('ul');
    this.dom_element.className = 'menu-list';
    const li = this.createFolderRepresentation(this.folders_dict, this.active_path);
    this.dom_element.appendChild(li);
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
}

export default FolderViewComponent;