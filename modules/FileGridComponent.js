class FileGridComponent {
  constructor({ containerId, filesData, renderPreviewContent, renderCustomData, renderActions, handleFileClick, numColumns = 5, minTileWidth = '20rem', onContextMenu = null, onMetaOpen = null}) {
    this.containerId = containerId; // ID of the HTML container to render the grid in
    this.filesData = filesData;     // Array of file data objects
    this.renderPreviewContent = renderPreviewContent; // Function to render preview content (customizable)
    this.renderCustomData = renderCustomData;   // Function to render custom data (customizable)
    this.renderActions = renderActions;       // Function to render actions (customizable)
    this.handleFileClick = handleFileClick;     // Function to handle file click events (customizable)
    this.numColumns = numColumns;             // Number of columns in the grid
    this.minTileWidth = minTileWidth;         // Minimum width of each tile
    this.onContextMenu = onContextMenu;       // Optional right-click handler (fileData, event, element)
    this.onMetaOpen = onMetaOpen;             // Optional meta open handler (fileData)
    // Derive a unique localStorage key from the container ID so each module
    // stores its own view mode independently.
    this._storageKey = 'fgc_view_' + containerId.replace(/^#/, '');
    this.viewMode = localStorage.getItem(this._storageKey) || 'grid';
    this.render(); // Initial rendering
  }

  render() {
    const container = $(this.containerId);
    container.empty();

    // ── View toggle toolbar ──────────────────────────────────────────
    const toolbar = document.createElement('div');
    toolbar.className = 'is-flex is-justify-content-flex-end mb-2';
    const btnGroup = document.createElement('div');
    btnGroup.className = 'buttons has-addons';
    ['grid', 'list'].forEach(mode => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'button is-small' + (this.viewMode === mode ? ' is-info is-selected' : '');
      btn.title = mode === 'grid' ? 'Grid view' : 'List view';
      btn.innerHTML = `<span class="icon"><i class="fas fa-${mode === 'grid' ? 'grip' : 'list'}"></i></span>`;
      btn.addEventListener('click', () => {
        this.viewMode = mode;
        localStorage.setItem(this._storageKey, mode);
        this.render();
      });
      btnGroup.appendChild(btn);
    });
    toolbar.appendChild(btnGroup);
    container.append(toolbar);

    if (!this.filesData || this.filesData.length === 0) {
      container.append('<p>No files to display.</p>');
      return;
    }

    if (this.viewMode === 'list') {
      const listContainer = document.createElement('div');
      listContainer.id = 'file_grid_inner_container';
      this.filesData.forEach(fileData => {
        listContainer.appendChild(this.createFileElement(fileData));
      });
      container.append(listContainer);
    } else {
      const gridHTML = `
        <div class="grid is-gap-0.5" id="file_grid_inner_container"
             style="grid-template-columns: repeat(auto-fill, minmax(${this.minTileWidth}, 1fr));">
        </div>`;
      container.append(gridHTML);
      const innerContainer = container.find('#file_grid_inner_container');
      this.filesData.forEach(fileData => {
        innerContainer.append(this.createFileElement(fileData));
      });
    }
  }

  createFileElement(fileData) {
    const isList = this.viewMode === 'list';

    const fileDiv = document.createElement('div');
    fileDiv.dataset.filePath = fileData.file_path;

    fileDiv.addEventListener('click', () => {
      if (this.handleFileClick) {
        this.handleFileClick(fileData);
      } else {
        console.log('File clicked:', fileData.file_path);
      }
    });

    fileDiv.addEventListener('contextmenu', (e) => {
      if (typeof this.onContextMenu === 'function') {
        e.preventDefault();
        this.onContextMenu(fileData, e, fileDiv);
      }
    });

    // ── Preview container (identical in both modes) ──────────────────
    const previewContainer = document.createElement('div');
    previewContainer.className = 'file-preview-container';
    previewContainer.style.aspectRatio = '1';
    previewContainer.style.display = 'flex';
    previewContainer.style.justifyContent = 'center';
    previewContainer.style.alignItems = 'center';
    previewContainer.style.position = 'relative';

    if (this.renderPreviewContent) {
      const previewContent = this.renderPreviewContent(fileData);
      previewContainer.appendChild(previewContent);
    }

    const externalMetaIsAvailable = !!(fileData && fileData.has_meta);
    console.log(`File: ${fileData.file_path}, has_meta: ${externalMetaIsAvailable}`);

    if (this.onMetaOpen && typeof this.onMetaOpen === 'function' && externalMetaIsAvailable) {
      const overlay = document.createElement('div');
      overlay.className = 'buttons is-flex is-align-items-center';
      overlay.style.position = 'absolute';
      overlay.style.left = '.35rem';
      overlay.style.bottom = '.35rem';
      overlay.style.zIndex = '2';
      overlay.style.display = 'none';

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'button is-small is-light is-rounded';
      btn.title = 'Edit .meta';
      btn.innerHTML = `<span class="icon"><i class="fas fa-file-pen"></i></span>`;
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.onMetaOpen(fileData);
      });

      overlay.appendChild(btn);
      previewContainer.appendChild(overlay);
    }

    // ── Custom data container (identical in both modes) ──────────────
    const customDataContainer = document.createElement('div');
    customDataContainer.className = 'file-custom-data';
    if (this.renderCustomData) {
      customDataContainer.appendChild(this.renderCustomData(fileData));
    }

    // ── Actions container (identical in both modes) ──────────────────
    const actionsContainer = document.createElement('div');
    if (this.renderActions) {
      actionsContainer.appendChild(this.renderActions(fileData));
    }

    // ── Layout ───────────────────────────────────────────────────────
    if (isList) {
      fileDiv.className = 'is-flex is-align-items-flex-start has-background-light p-2 mb-1 is-clickable';

      // Preview column: same square container, same width as a grid tile
      const previewWrap = document.createElement('div');
      previewWrap.style.cssText = `width:${this.minTileWidth}; flex-shrink:0;`;
      previewWrap.appendChild(previewContainer);

      // Custom data column: fixed width, horizontally separated
      const customWrap = document.createElement('div');
      customWrap.className = 'mx-3';
      customWrap.style.cssText = `width:${this.minTileWidth}; flex-shrink:0;`;
      customWrap.appendChild(customDataContainer);
      customWrap.appendChild(actionsContainer);

      // Description column: takes remaining space, shows preview_text
      const descWrap = document.createElement('div');
      descWrap.className = 'is-flex-grow-1';
      descWrap.style.minWidth = '0';
      const previewText = (fileData.file_info || {}).preview_text || '';
      if (previewText) {
        const p = document.createElement('p');
        p.style.cssText = 'overflow:hidden; display:-webkit-box; -webkit-line-clamp:12; -webkit-box-orient:vertical;';
        p.textContent = previewText;
        descWrap.appendChild(p);
      }

      fileDiv.appendChild(previewWrap);
      fileDiv.appendChild(customWrap);
      fileDiv.appendChild(descWrap);
    } else {
      fileDiv.className = 'cell has-background-light p-1 is-flex is-flex-direction-column is-justify-content-space-between is-clickable';
      fileDiv.appendChild(previewContainer);
      fileDiv.appendChild(customDataContainer);
      fileDiv.appendChild(actionsContainer);
    }

    return fileDiv;
  }

  updateFiles(newFilesData) {
    this.filesData = newFilesData;
    this.render();
  }
}

export default FileGridComponent;