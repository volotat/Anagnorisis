class FileGridComponent {
  constructor({ containerId, filesData, renderPreviewContent, renderCustomData, renderActions, handleFileClick, numColumns = 5, minTileWidth = '20rem'}) {
    this.containerId = containerId; // ID of the HTML container to render the grid in
    this.filesData = filesData;     // Array of file data objects
    this.renderPreviewContent = renderPreviewContent; // Function to render preview content (customizable)
    this.renderCustomData = renderCustomData;   // Function to render custom data (customizable)
    this.renderActions = renderActions;       // Function to render actions (customizable)
    this.handleFileClick = handleFileClick;     // Function to handle file click events (customizable)
    this.numColumns = numColumns;             // Number of columns in the grid
    this.minTileWidth = minTileWidth;         // Minimum width of each tile
    this.render(); // Initial rendering
  }

  render() {
    const container = $(this.containerId);
    container.empty(); // Clear any existing content

    if (!this.filesData || this.filesData.length === 0) {
      container.html('<p>No files to display.</p>'); // Handle empty data case
      return;
    }
    
    const gridHTML = `
      <div class="grid is-gap-0.5" id="file_grid_inner_container"
           style="
             grid-template-columns: repeat(auto-fill, minmax(${this.minTileWidth}, 1fr)); 
           ">
      </div>`;
    container.append(gridHTML);
    const innerContainer = container.find('#file_grid_inner_container');

    this.filesData.forEach(fileData => {
      const fileElement = this.createFileElement(fileData);
      innerContainer.append(fileElement);
    });
  }

  createFileElement(fileData) {
    const fileDiv = document.createElement('div');
    fileDiv.className = 'cell has-background-light p-1 is-flex is-flex-direction-column is-justify-content-space-between is-clickable'; // Added is-clickable class
    fileDiv.dataset.filePath = fileData.file_path; // Store file path in dataset
    fileDiv.addEventListener('click', () => {  // Attach click handler
      if (this.handleFileClick) {
        this.handleFileClick(fileData); // Call the custom click handler
      } else {
        console.log('File clicked:', fileData.file_path); // Default action if no handler provided
      }
    });


    const previewContainer = document.createElement('div');
    previewContainer.className = 'file-preview-container'; // Add class for styling
    previewContainer.style.aspectRatio = '1'; // Make preview container square
    previewContainer.style.display = 'flex';
    previewContainer.style.justifyContent = 'center';
    previewContainer.style.alignItems = 'center';
    //previewContainer.style.maxWidth = '300px';
    fileDiv.appendChild(previewContainer);

    // Render Preview Content (using customizable function)
    if (this.renderPreviewContent) {
      const previewContent = this.renderPreviewContent(fileData);
      previewContainer.appendChild(previewContent);
    }

    // Custom Data Display Container
    const customDataContainer = document.createElement('div');
    customDataContainer.className = 'file-custom-data';
    fileDiv.appendChild(customDataContainer);

    // Render Custom Data (using customizable function)
    if (this.renderCustomData) {
      const customDataContent = this.renderCustomData(fileData);
      customDataContainer.appendChild(customDataContent);
    }

    // Create a level container for buttons and checkbox (actions) 
    const actionsContainer = document.createElement('div');
    //actionsContainer.className = 'level is-gapless mt-1'; // Added mt-1 for spacing
    fileDiv.appendChild(actionsContainer);

    if (this.renderActions) {
      const customActions = this.renderActions(fileData);
      actionsContainer.appendChild(customActions);
    }


    return fileDiv;
  }

  updateFiles(newFilesData) {
    this.filesData = newFilesData;
    this.render(); // Re-render the grid with new data
  }
}

export default FileGridComponent;