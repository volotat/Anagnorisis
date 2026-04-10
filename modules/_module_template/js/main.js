/**
 * main.js — Module frontend logic (REQUIRED)
 *
 * This file is loaded as an ES module from page.html.
 * The global `socket` variable (Socket.IO client) is available from base.html.
 *
 * Shared UI components are importable from /modules/:
 *   - SearchBarComponent  — text input + sorting/mode dropdown + temperature control
 *   - FileGridComponent   — responsive grid of file cards
 *   - PaginationComponent — page navigation controls
 *   - FolderViewComponent — collapsible folder tree sidebar
 *   - StarRatingComponent — clickable 1–10 star widget
 *   - ContextMenuComponent — right-click context menu builder
 *   - MetaEditor          — file metadata editor modal (editable + read-only modes)
 *
 * TaskManagerComponent is auto-initialised by base.html — no import needed.
 *
 * Naming conventions for socket events:
 *   emit_{module}_page_{action}   — must match the handlers in serve.py
 */

import SearchBarComponent  from '/modules/SearchBarComponent.js';
import FileGridComponent   from '/modules/FileGridComponent.js';
import PaginationComponent from '/modules/PaginationComponent.js';
import FolderViewComponent from '/modules/FolderViewComponent.js';
import StarRatingComponent from '/modules/StarRating.js';
import ContextMenuComponent from '/modules/ContextMenuComponent.js';
import MetaEditor          from '/modules/MetaEditor.js';

// Wrap everything in an IIFE to avoid polluting the global scope
(function () {
  // ── Constants ──────────────────────────────────────────────────────
  const MODULE_NAME = 'example';                       // Must match serve.py module_name
  const FILES_PER_PAGE = 24;
  const FILES_PER_ROW = 6;

  // ── URL state ──────────────────────────────────────────────────────
  const urlParams = new URLSearchParams(window.location.search);
  let pageParam = parseInt(urlParams.get('page'));
  let page = (!pageParam || pageParam < 1) ? 1 : pageParam;
  let path = decodeURIComponent(urlParams.get('path') || '');

  // Stable random seed (preserved across page changes)
  let seed = urlParams.get('seed');
  if (!seed) {
    seed = Math.floor(Math.random() * 1e9);
    urlParams.set('seed', seed);
    window.history.replaceState({}, '', '?' + urlParams.toString());
  }

  // ── MetaEditor instances ───────────────────────────────────────────
  // Editable instance — for .meta sidecar files
  const metaEditor = new MetaEditor({
    containerId: 'meta_editor_container',
    readOnly: false,
    onSave: (filePath, content) => {
      socket.emit(`emit_${MODULE_NAME}_page_save_external_metadata_file_content`, {
        file_path: filePath,
        metadata_content: content,
      });
    },
  });

  // Read-only instance — for viewing full AI-generated descriptions
  const fullDescriptionViewer = new MetaEditor({
    containerId: 'full_description_container',
    readOnly: true,
  });

  // ── Context menu ───────────────────────────────────────────────────
  const contextMenu = new ContextMenuComponent();

  function createContextMenuForFile(fileData) {
    return [
      {
        label: 'Open in new tab',
        action: () => {
          window.open(`/${MODULE_NAME}_files/${fileData.file_path}`, '_blank');
        },
      },
      {
        label: 'Find similar',
        action: () => {
          searchBar.setValue(fileData.file_path);
          searchBar.setSortBy('similarity');
          page = 1;
          requestFiles();
        },
      },
      {
        label: 'Edit .meta file',
        action: () => {
          socket.emit(`emit_${MODULE_NAME}_page_get_external_metadata_file_content`,
            fileData.file_path, (response) => {
              metaEditor.open(response.file_path, response.content);
            });
        },
      },
      {
        label: 'Show full search description',
        action: () => {
          socket.emit(`emit_${MODULE_NAME}_page_get_full_metadata_description`,
            fileData.file_path, (response) => {
              fullDescriptionViewer.open(response.file_path, response.content);
            });
        },
      },
    ];
  }

  // ── Render helpers ─────────────────────────────────────────────────

  /**
   * Called by FileGridComponent for each file to produce the preview element.
   * Return an HTMLElement that will be placed inside the grid card.
   *
   * Adapt this for your media type — e.g. <img>, <audio>, <video>, text preview.
   */
  function renderPreview(fileData) {
    const el = document.createElement('div');
    el.className = 'has-background-light p-3';
    el.style.aspectRatio = '1';
    el.textContent = fileData.file_info.base_name || fileData.file_path;
    return el;
  }

  /**
   * Called by FileGridComponent to render extra metadata below the preview.
   * Return an HTMLElement (or null to skip).
   */
  function renderCustomData(fileData) {
    const container = document.createElement('div');
    container.style.wordBreak = 'break-word';

    if (fileData.search_score != null) {
      const score = document.createElement('p');
      score.innerHTML = `<b>Score:</b>&nbsp;${fileData.search_score.toFixed(3)}`;
      container.appendChild(score);
    }

    const pathEl = document.createElement('p');
    pathEl.innerHTML = `<b>Path:</b>&nbsp;${fileData.file_path}`;
    container.appendChild(pathEl);

    const sizeEl = document.createElement('p');
    sizeEl.innerHTML = `<b>Size:</b>&nbsp;${fileData.file_info.file_size || 'N/A'}`;
    container.appendChild(sizeEl);

    return container;
  }

  // ── Search bar ─────────────────────────────────────────────────────
  const searchBar = new SearchBarComponent({
    containerId: 'search_bar_container',
    moduleName: MODULE_NAME,
    socket: socket,
    // Available sort options shown in the dropdown:
    sortOptions: [
      { value: 'by_text',    label: 'Relevance' },
      { value: 'file_size',  label: 'File size' },
      { value: 'random',     label: 'Random' },
      { value: 'rating',     label: 'Rating' },
    ],
    onSearch: (query, sortBy) => {
      page = 1;
      requestFiles();
    },
  });

  // ── Folder tree ────────────────────────────────────────────────────
  const folderView = new FolderViewComponent({
    containerId: 'folder_tree_container',
    onFolderSelect: (folderPath) => {
      path = folderPath;
      page = 1;
      requestFiles();
    },
  });

  // Request folder tree from backend
  socket.emit(`emit_${MODULE_NAME}_page_get_folders`, { path: '' }, (data) => {
    folderView.render(data);
  });

  // ── Pagination ─────────────────────────────────────────────────────
  const pagination = new PaginationComponent({
    containerSelectors: ['.pagination-list'],
    onPageChange: (newPage) => {
      page = newPage;
      requestFiles();
    },
  });

  // ── File grid ──────────────────────────────────────────────────────
  const fileGrid = new FileGridComponent({
    containerId: 'example_grid_container',
    columnsPerRow: FILES_PER_ROW,
    renderPreview: renderPreview,
    renderCustomData: renderCustomData,
    onContextMenu: (fileData, event) => {
      event.preventDefault();
      contextMenu.show(event, createContextMenuForFile(fileData));
    },
  });

  // ── Data fetching ──────────────────────────────────────────────────

  function requestFiles() {
    const { query, sortBy } = searchBar.getValues();

    socket.emit(`emit_${MODULE_NAME}_page_get_files`, {
      path: path,
      pagination: page,
      limit: FILES_PER_PAGE,
      text_query: query,
      seed: seed,
      sort_by: sortBy,
    });
  }

  // Listen for the file list response from the server
  socket.on(`emit_${MODULE_NAME}_page_show_files`, (data) => {
    fileGrid.render(data.files);
    pagination.update(page, data.total_pages);

    // Update status text
    document.querySelectorAll(`.${MODULE_NAME}-search-status`).forEach((el) => {
      el.textContent = `${data.total_files} files`;
    });
  });

  // Listen for generic search status updates
  socket.on('emit_show_search_status', (msg) => {
    document.querySelectorAll(`.${MODULE_NAME}-search-status`).forEach((el) => {
      el.textContent = msg;
    });
  });

  // ── Initial load ───────────────────────────────────────────────────
  requestFiles();
})();
