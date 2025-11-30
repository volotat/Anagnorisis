import FolderViewComponent from '/pages/FolderViewComponent.js';
import StarRatingComponent from '/pages/StarRating.js';
import PlaylistManager from '/pages/music/js/PlaylistManager.js';
import SongControlPanel from '/pages/music/js/SongControlPanel.js';
import PaginationComponent from '/pages/PaginationComponent.js';
import FileGridComponent from '/pages/FileGridComponent.js';
import SearchBarComponent from '/pages/SearchBarComponent.js';
import ContextMenuComponent from '/pages/ContextMenuComponent.js';
import MetaEditor from '/pages/MetaEditor.js';

//// BEFORE PAGE LOADED

function renderImagePreview(fileData) { // Function for Images module preview
    const imageDataDiv = document.createElement('div');
    imageDataDiv.className = 'cell p-1 is-flex is-flex-direction-column is-justify-content-space-between';

    const imageContainer = document.createElement('div');
    imageContainer.classList.add('pswp-gallery__item', 'is-flex-direction-column');
    // Make image container always square
    imageContainer.style.aspectRatio = 1;
    // Horizontally center the image
    imageContainer.style.display = 'flex';
    imageContainer.style.justifyContent = 'center'; 
    imageContainer.style.alignItems = 'center';
    $(imageContainer).addClass('mb-2');


    imageDataDiv.append(imageContainer);

    // Create an object for holding the image data
    if (fileData.file_info.audiofile_data['image'] == null){
      // Create an icon with music note from font awesome
      const icon = document.createElement('span');
      icon.className = 'icon is-large is-fullwidth';
      icon.innerHTML = '<i class="fas fa-6x fa-music"></i>';
      imageContainer.append(icon);
    } else {
      const image = document.createElement('img');
      image.src = fileData.file_info.audiofile_data['image']; 
      imageContainer.append(image);
    }

    return imageDataDiv;
}

function renderCustomData(fileData) { // Function for custom data rendering
    const dataContainer = document.createElement('div');
    dataContainer.className = 'file-custom-data';
    dataContainer.style.wordBreak = 'break-word';

    // Search matching scores
    if (fileData.search_score !== null && fileData.search_score !== undefined) {
        const searchScoresElement = document.createElement('p');
        searchScoresElement.className = 'file-info file-search-scores';
        searchScoresElement.innerHTML = `<b>Search Score:</b>&nbsp;${(fileData.search_score || 0).toFixed(3)}`;
        dataContainer.appendChild(searchScoresElement);
    }
    
    // File Path
    const filePathElement = document.createElement('p');
    filePathElement.className = 'file-info file-path';
    filePathElement.innerHTML = `<b>Path:</b>&nbsp;${fileData.file_path}`;
    dataContainer.appendChild(filePathElement);

    // Hash
    const hashElement = document.createElement('p');
    hashElement.className = 'file-info file-hash';
    hashElement.innerHTML = `<b>Hash:</b>&nbsp;${fileData.hash}`;
    dataContainer.appendChild(hashElement);

    // File Size
    const fileSizeElement = document.createElement('p');
    fileSizeElement.className = 'file-info file-size';
    fileSizeElement.innerHTML = `<b>Size:</b>&nbsp;${fileData.file_size}`;
    dataContainer.appendChild(fileSizeElement);

    // User Rating
    const userRatingElement = document.createElement('p');
    userRatingElement.className = 'file-info file-user-rating';
    userRatingElement.innerHTML = `<b>User rating:</b>&nbsp;${fileData.file_info.user_rating !== null ? fileData.file_info.user_rating : 'N/A'}`;
    dataContainer.appendChild(userRatingElement);

    // Model Rating
    const modelRatingElement = document.createElement('p');
    modelRatingElement.className = 'file-info file-model-rating';
    modelRatingElement.innerHTML = `<b>Model rating:</b>&nbsp;${fileData.file_info.model_rating !== null ? Math.round(fileData.file_info.model_rating * 100) / 100 : 'N/A'}`;
    dataContainer.appendChild(modelRatingElement);

    // Length
    const lengthElement = document.createElement('p');
    lengthElement.className = 'file-info file-length';
    lengthElement.innerHTML = `<b>Length:</b>&nbsp;${fileData.file_info.length}`;
    dataContainer.appendChild(lengthElement);

    // Last Played
    const lastPlayedElement = document.createElement('p');
    lastPlayedElement.className = 'file-info file-last-played';
    lastPlayedElement.innerHTML = `<b>Last played:</b>&nbsp;${fileData.file_info.last_played}`;
    dataContainer.appendChild(lastPlayedElement);

    return dataContainer;
}

// Create a closed scope to avoid any variable collisions  
(function() {
  //// CONSTANTS AND VARIABLES
  let num_files_on_page = 24;
  let num_files_in_row = 6; // TODO: calculate from the screen size
  let selected_files = [];
  let all_files_paths = [];
  let playlistManager;
  let songControlPanel;

  //// BEFORE PAGE LOADED
  // read page number from the URL ?page=1
  const urlParams = new URLSearchParams(window.location.search);
  let pageParam = parseInt(urlParams.get('page'));
  let page = (!pageParam || pageParam < 1) ? 1 : pageParam;
  // let text_query = urlParams.get('text_query') || '';
  let path = urlParams.get('path') || '';
  // text_query = decodeURIComponent(text_query);
  path = decodeURIComponent(path);
  console.log('path', path);

  // Create generic .meta editor wired to Images socket events
  const externalMetaEditor = new MetaEditor({
    api: {
      // Load .meta (Images backend expects the file_path string, returns {content, file_path})
      load: (filePath, onLoaded) => {
        socket.emit('emit_music_page_get_external_metadata_file_content', filePath, (response)=>{
          onLoaded(response.content || '');
        });
      },
      // Save .meta
      save: async (filePath, content) => {
        socket.emit('emit_music_page_save_external_metadata_file_content', {
          file_path: filePath,
          metadata_content: content
        });
      }
    }
  });  
  function openExternalMetaEditorForFile(fileData, readOnly=false) {
    externalMetaEditor.open({
      filePath: fileData.file_path,                        // relative path inside media dir
      displayName: (fileData.base_name + '.meta') || '',   // optional nice title
    });
  }


  const fullDescriptionMetaEditor = new MetaEditor({
    api: {
      // Load full description (Music backend expects the file_path string, returns {content, file_path})
      load: (filePath, onLoaded) => {
        socket.emit('emit_music_page_get_full_metadata_description', filePath, (response)=>{
          onLoaded(response.content || '');
        });
      },
      // Save full description (not implemented)
      save: (filePath, content) => {
        // No saving for full description
        return Promise.resolve();
      }
    },
    readOnly: true, // Always read-only
  });
  
  function openFullDescriptionForFile(fileData) {
    fullDescriptionMetaEditor.open({
      filePath: fileData.file_path,                                     // relative path inside media dir
      displayName: (fileData.base_name + ' full search description') || '',   // optional nice title
    });
  }

  // Create context menu for file items
  const ctxMenu = new ContextMenuComponent();
  function createContextMenuForFile(fileData, event) {
    ctxMenu.show(event.pageX, event.pageY, [
      {
        label: 'Open in new tab',
        action: () => {
          window.open('music_files/'+fileData.file_path, '_blank');
        }
      },
      {
        label: 'Find similar music',
        action: () => {
          let url = new URL(window.location.href);
          let params = new URLSearchParams(url.search);
          params.set('text_query', fileData.full_path);
          params.set('page', 1);
          params.set('mode', 'semantic-content');
          url.search = params.toString();
          window.location.href = url.toString();
        }
      },
      {
        label: 'Edit internal metadata',
        action: () => {
          alert('Not yet implemented action: "Edit internal metadata"');
        }
      },
      {
        label: 'Edit .meta file',
        icon: 'fas fa-file-pen',
        action: () => { 
          openExternalMetaEditorForFile(fileData); 
        }
      },
      {
        label: 'Show full search description',
        icon: 'fas fa-info-circle',
        action: () => { 
          openFullDescriptionForFile(fileData);
        }
      },
      { type: 'divider'},
      {
        label: 'Rename',
        icon: 'fas fa-edit',
        action: () => {
          alert('Not yet implemented action: "Rename"');
        }
      },
      {
        label: 'Move to...',
        icon: 'fas fa-file-import',
        action: () => {
          alert('Not yet implemented action: "Move to..."');
        }
      },
      { label: 'Delete',
        icon: 'fas fa-trash',
        action: () => {
          alert('Not yet implemented action: "Delete"');
        }
      }
    ]);
  }
  
  //// AFTER PAGE LOADED
  $(document).ready(function() {
    // Set up audio elements
    const audioPlayer = $('#audio_player')[0];
    const playlistElement = $('#playlist_content');
    const song_cover_image_element =  $('#song_cover_image');
    const song_progress_element =  $('#song_progress');
    const song_label_element = $('#song_label');
    const song_rating_element = $('#song_rating');

    let paginationComponent; // Declare paginationComponent in the scope 

    // Instantiate SearchBarComponent
    const searchBar = new SearchBarComponent({
      container: '#search_bar_container',
      enableModes: ['file-name', 'semantic-content', 'semantic-metadata'], // disable here as needed
      showOrder: true,
      showTemperature: true,
      temperatures: [0, 0.2, 1, 2],
      keywords: ['recommendation', 'rating', 'random', 'file_size', 'length', 'similarity'],
      autoSyncUrl: true,
      ensureDefaultsInUrl: true,
    });

    const search_state = searchBar.getState();
    
    playlistManager = new PlaylistManager(
      audioPlayer, 
      playlistElement, 
      song_cover_image_element, 
      song_label_element, 
      socket
    );
    songControlPanel = new SongControlPanel(
      audioPlayer,
      song_cover_image_element,
      song_label_element,
      song_rating_element,
      song_progress_element,
      socket,
      playlistManager
    );
    playlistManager.songControlPanel = songControlPanel;

    // Load the playlist and song status
    const savedPlaylist = JSON.parse(localStorage.getItem("music_page_playlist")); 
    const savedPlayTime = parseFloat(localStorage.getItem("music_page_song_play_time"));
    const savedCurrentSongIndex = parseInt(localStorage.getItem("music_page_current_song_index"));
    const savedIsPlaying = localStorage.getItem("music_page_is_playing");

    if(savedPlaylist && savedCurrentSongIndex != null) {
      // load saved playlist and play current song index
      playlistManager.setPlaylist(savedPlaylist);
      playlistManager.playSongAtIndex(savedCurrentSongIndex);

      // Set playback status
      audioPlayer.currentTime = savedPlayTime || 0;

      $("#song_control_panel_dectivated").addClass('is-hidden');
      $("#song_control_panel_activated").removeClass('is-hidden');

      if(savedIsPlaying == 'playing') {
        songControlPanel.playSong();
      } else {
        songControlPanel.pauseSong();
      }
    }


    // Request current media folder path
    socket.emit('emit_music_page_get_path_to_media_folder');

    // Request files from the main media folder
    // --- Folder View ---
    socket.emit('emit_music_page_get_folders', {
      path: path, 
    }, (response) => { // event handler for folders
      console.log('emit_music_page_get_folders', response);

      const folderView = new FolderViewComponent(response.folders, response.folder_path, false); // Enable context menu
      $('.menu').append(folderView.getDOMElement());
    });
    
    // --- File List ---
    socket.emit('emit_music_page_get_files', {
      path: path, 
      pagination: (page-1)*num_files_on_page, 
      limit: num_files_on_page,
      text_query: search_state.text_query,
      seed: search_state.seed,
      mode: search_state.mode,
      order: search_state.order,
      temperature: search_state.temperature,
    }, (response) => {
      console.log('emit_music_page_get_files response', response);

      all_files_paths = response["all_files_paths"];

      // Update or Initialize FileGridComponent
      const fileGridComponent = new FileGridComponent({
          containerId: '#music_grid_container', // Use the new container ID
          filesData: response.files_data,
          renderPreviewContent: renderImagePreview, // Pass the text preview function
          renderCustomData: renderCustomData, // Pass the custom data rendering function
          // renderActions: renderActions, // Pass the actions rendering function
          handleFileClick: (fileData) => {},
          numColumns: num_files_in_row, 
          onContextMenu: createContextMenuForFile,
          onMetaOpen: openExternalMetaEditorForFile,
      });
      // Highlight playing song if any
      if (playlistManager.playlist.length > 0) {
        const currentSong = playlistManager.playlist[playlistManager.currentSongIndex];
        $(document).trigger('music:playing', [currentSong.file_path]);
      }

      // Update or Initialize Pagination Component
      const paginationContainer = $('.pagination.is-rounded.level-left.mb-0 .pagination-list'); // Select pagination container
      const urlParams = new URLSearchParams(window.location.search); // Get URL parameters for pattern
      const urlPattern = `?${urlParams.toString()}`;

      if (!paginationComponent) { // Instantiate PaginationComponent if it doesn't exist yet
          paginationComponent = new PaginationComponent({
              containerId: paginationContainer.closest('.pagination').get(0), // Pass the pagination nav element
              currentPage: page,
              totalPages: Math.ceil(response["total_files"] / num_files_on_page),
              urlPattern: urlPattern,
          });
      } else { // Update existing PaginationComponent
          paginationComponent.updatePagination(page, Math.ceil(response["total_files"] / num_files_on_page));
      }

      // Activate "Update Playlist" buttons
      $('.update_playlist').prop('disabled', false);
    });

    // Display current search status
    socket.on('emit_show_search_status', (status) => {
      $('.music-search-status').html(status);
    });

    // Show current media folder path
    socket.on('emit_music_page_show_path_to_media_folder', (current_path) => {
      $('#path_to_media_folder').val(current_path);
    });

    // Update path to the media folder
    $(`#update_path_to_media_folder`).click(()=>{ 
      socket.emit('emit_music_page_update_path_to_media_folder', $('#path_to_media_folder').val());
      // refresh page after a second
      setTimeout(function(){
        location.reload();
      }, 500);
    });
    
    // Update the playlist
    $('.update_playlist').click(function() {
      if (selected_files.length > 0) {
        playlistManager.setPlaylist(selected_files);
      } else {
        playlistManager.setPlaylist(all_files_paths);
      }

      playlistManager.playSongAtIndex(0);
      songControlPanel.updateButtons();

      $("#song_control_panel_dectivated").addClass('is-hidden');
      $("#song_control_panel_activated").removeClass('is-hidden');
    });

    // Unselect all files
    $('#unselect_all_files').click(function() {
      $("input[type='checkbox']").prop('checked', false);
      $('#files_actions').hide();
      selected_files = [];
    });

    // Slightly highlight the currently playing song in the grid
    let lastHighlightedTile = null;
    $(document).on('music:playing', (_e, filePath) => {
      console.log('Highlighting playing song tile:', filePath);

      const container = document.querySelector('#music_grid_container');
      if (!container) return;

      // Remove highlight from the previously active tile
      if (lastHighlightedTile) {
        lastHighlightedTile.classList.remove('has-background-info-light', 'is-current-playing');
        lastHighlightedTile.classList.add('has-background-light');
        lastHighlightedTile = null;
      }

      // Find the tile by dataset (safe for special chars like [ ])
      const tiles = container.querySelectorAll('[data-file-path]');
      for (const tile of tiles) {
        if (tile.dataset.filePath === filePath) {
          tile.classList.remove('has-background-light');
          tile.classList.add('has-background-info-light', 'is-current-playing');
          lastHighlightedTile = tile;
          break;
        }
      }
    });
  })

  //// RESPONDS TO SOCKET EVENTS
  //socket.on('emit_music_page_add_radio_state', (state) => {
  //  add_radio_state(state)
  //});
})();