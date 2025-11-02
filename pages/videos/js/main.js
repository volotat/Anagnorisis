import FolderViewComponent from '/pages/FolderViewComponent.js';
import FileGridComponent from '/pages/FileGridComponent.js';
import PaginationComponent from '/pages/PaginationComponent.js';
import SearchBarComponent from '/pages/SearchBarComponent.js';
import ContextMenuComponent from '/pages/ContextMenuComponent.js';
import MetaEditor from '/pages/MetaEditor.js';

// Create a closed scope to avoid any variable collisions  
(function() {
  //// CONSTANTS AND VARIABLES
  let num_files_on_page = 20;

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

  // function resize_image(image, maxWidth, maxHeight) {
  //   const canvas = document.createElement('canvas');
  //   const ctx = canvas.getContext('2d');

  //   // Set the dimensions of the canvas to the desired dimensions of the image
  //   let width = image.width;
  //   let height = image.height;

  //   // Calculate the width and height, maintaining the aspect ratio
  //   if (width > height) {
  //     if (width > maxWidth) {
  //       height *= maxWidth / width;
  //       width = maxWidth;
  //     }
  //   } else {
  //     if (height > maxHeight) {
  //       width *= maxHeight / height;
  //       height = maxHeight;
  //     }
  //   }

  //   // Set the canvas width and height and draw the image data into the canvas
  //   canvas.width = width;
  //   canvas.height = height;
  //   ctx.drawImage(image, 0, 0, width, height);

  //   // Get the reduced image data from the canvas
  //   const reducedImage = new Image();
  //   reducedImage.src = canvas.toDataURL();

  //   return reducedImage;
  // }

  function renderVideoPreview(fileData) {
    const videoPreviewContainer = document.createElement('div');
    videoPreviewContainer.className = 'video-preview-container';
    videoPreviewContainer.style.aspectRatio = '16 / 9';
    videoPreviewContainer.style.display = 'flex';
    videoPreviewContainer.style.justifyContent = 'center';
    videoPreviewContainer.style.alignItems = 'center';
    videoPreviewContainer.style.overflow = 'hidden'; // Ensure image doesn't overflow

    const image = document.createElement('img');
    image.src = '/video_files/' + fileData.file_info.preview_path; // Use preview_path
    image.style.maxWidth = '100%';
    image.style.maxHeight = '100%';
    image.style.objectFit = 'contain';
    videoPreviewContainer.appendChild(image);
    return videoPreviewContainer;
  }

  function renderCustomData(fileData) {
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

    // User Rating
    const userRatingElement = document.createElement('p');
    userRatingElement.className = 'file-info file-user-rating';
    userRatingElement.innerHTML = `<b>User rating:</b>&nbsp;${fileData.file_info.user_rating !== null ? fileData.file_info.user_rating.toFixed(1) : 'N/A'}`;
    dataContainer.appendChild(userRatingElement);

    // Model Rating
    const modelRatingElement = document.createElement('p');
    modelRatingElement.className = 'file-info file-model-rating';
    modelRatingElement.innerHTML = `<b>Model rating:</b>&nbsp;${fileData.file_info.model_rating !== null ? fileData.file_info.model_rating.toFixed(1) : 'N/A'}`;
    dataContainer.appendChild(modelRatingElement);

    // File Size
    const fileSizeElement = document.createElement('p');
    fileSizeElement.className = 'file-info file-size';
    fileSizeElement.innerHTML = `<b>Size:</b>&nbsp;${fileData.file_size}`;
    dataContainer.appendChild(fileSizeElement);

    // Length
    const lengthElement = document.createElement('p');
    lengthElement.className = 'file-info file-length';
    lengthElement.innerHTML = `<b>Length:</b>&nbsp;${fileData.file_info.length}`;
    dataContainer.appendChild(lengthElement);
    
    // Last Played
    const lastPlayedElement = document.createElement('p');
    lastPlayedElement.className = 'file-info file-last-played';
    lastPlayedElement.innerHTML = `<b>Last Played:</b>&nbsp;${fileData.file_info.last_played}`;
    dataContainer.appendChild(lastPlayedElement);

    return dataContainer;
  }

  // Create generic .meta editor wired to Images socket events
  const metaEditor = new MetaEditor({
    api: {
      // Load .meta (Images backend expects the file_path string, returns {content, file_path})
      load: (filePath, onLoaded) => {
        socket.emit('emit_videos_page_get_external_metadata_file_content', filePath, (response)=>{
          onLoaded(response.content || '');
        });
      },
      // Save .meta
      save: async (filePath, content) => {
        socket.emit('emit_videos_page_save_external_metadata_file_content', {
          file_path: filePath,
          metadata_content: content
        });
      }
    }
  });  
  function openMetaEditorForFile(fileData) {
    metaEditor.open({
      filePath: fileData.file_path,           // relative path inside media dir
      displayName: fileData.base_name || ''   // optional nice title
    });
  }

  // Create context menu for file items
  const ctxMenu = new ContextMenuComponent();
  function createContextMenuForFile(fileData, event) {
    ctxMenu.show(event.pageX, event.pageY, [
      {
        label: 'Open in new tab',
        action: () => {
          window.open('video_files/'+fileData.file_path, '_blank');
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
        action: () => { openMetaEditorForFile(fileData); }
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
    let paginationComponent;

    const modal = document.getElementById('video-modal');
    const openBtn = document.getElementById('open-video-modal');
    const closeBtn = modal.querySelector('.modal-close');
    const modalBg = modal.querySelector('.modal-background');
    const video = document.getElementById('modal-video-player');

    // Instantiate SearchBarComponent
    const searchBar = new SearchBarComponent({
      container: '#search_bar_container',
      enableModes: ['file-name', 'semantic-metadata'], // disable here as needed
      showOrder: true,
      showTemperature: true,
      temperatures: [0, 0.2, 1, 2],
      keywords: ['recommendation',  'random'],
      autoSyncUrl: true,
      ensureDefaultsInUrl: true,
    });

    const search_state = searchBar.getState();

    video.addEventListener('ended', () => {
      const list = window.videoPlaylist || [];
      const next = list[(window.currentVideoIndex || 0) + 1];
      if (!next) return;
      window.currentVideoIndex += 1;
      socket.emit('emit_videos_page_start_streaming', next.file_path, (response) => {
        if (response && response.stream_url) {
          if (window.currentHls) {
            video.currentTime = 0;
            window.currentHls.loadSource(response.stream_url);
            window.currentHls.attachMedia(video);
          } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
            video.src = response.stream_url;
          }
          socket.emit('emit_videos_page_video_start_playing', next.hash);
          video.play().catch(()=>{});
        }
      });
    });

    // Request files from the main media folder
    // --- Folder View ---
    socket.emit('emit_videos_page_get_folders', {
        path: path, 
    }, (response) => { // event handler for folders
        console.log('emit_videos_page_get_folders', response);

        const folderView = new FolderViewComponent(response.folders, response.folder_path, false); // Disable context menu
        $('.menu').append(folderView.getDOMElement());
    });

    // --- File List ---
    let currentFilePath = null; // To store currently opened file path
    socket.emit('emit_videos_page_get_files', {
      path: path, 
      pagination: (page-1)*num_files_on_page, 
      limit: num_files_on_page,
      text_query: search_state.text_query,
      seed: search_state.seed,
      mode: search_state.mode,
      order: search_state.order,
      temperature: search_state.temperature,
    }, (response) => {
        console.log('emit_text_page_show_files', response);
        window.videoPlaylist = response.files_data;

        // Update or Initialize FileGridComponent
        const fileGridComponent = new FileGridComponent({
            containerId: '#videos_files_grid_container', // Use the new container ID
            filesData: response.files_data,
            renderPreviewContent: renderVideoPreview, // Pass the preview function
            renderCustomData: renderCustomData, // No custom data rendering for now
            handleFileClick: handleFileClickAction,
            minTileWidth: '22rem', // Minimum width for each tile
            onContextMenu: createContextMenuForFile,
            onMetaOpen: openMetaEditorForFile,
        });

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

        // Close modal functions
        const closeModal = () => {
          modal.classList.remove('is-active');
          
          // 1. Stop video playback
          video.pause();
          
          // 2. Destroy HLS player if it exists
          if (window.currentHls) {
            window.currentHls.destroy();
            window.currentHls = null;
          }
          
          // 3. Clear video source
          video.src = '';
          video.load();
          
          // 4. Hide loading indicator if visible
          if (document.getElementById('loading-indicator')) {
            document.getElementById('loading-indicator').style.display = 'none';
          }
          
          // 5. Notify server to stop transcoding and clean up resources
          if (window.currentStreamId) {
            socket.emit('emit_videos_page_stop_streaming', window.currentStreamId, (response) => {
              console.log('Stream cleanup response:', response);
              window.currentStreamId = null;
            });
          }
        };

        // Close when X button clicked
        closeBtn.addEventListener('click', closeModal);
        
        // Close when clicking outside the modal
        modalBg.addEventListener('click', closeModal);
    });

    const handleFileClickAction = (fileData) => {
        //Logic to open modal and play video
        modal.classList.add('is-active');
        window.currentVideoIndex = (window.videoPlaylist || []).findIndex(f => f.hash === fileData.hash);
        
        // Request a stream for the video file
        socket.emit('emit_videos_page_start_streaming', fileData.file_path, (response) => {
            if (response && response.stream_url) {
                // Store the stream ID for cleanup later
                window.currentStreamId = response.stream_id;
                // Emit event to update last_played timestamp for the video
                socket.emit('emit_videos_page_video_start_playing', fileData.hash); 

                // Use an HLS.js player
                if (Hls.isSupported()) {
                    const video = document.getElementById('modal-video-player');
                    // Destroy previous HLS instance if it exists
                    if (window.currentHls) {
                        window.currentHls.destroy();
                    }
                    const hls = new Hls({
                        maxBufferLength: 60,           // Increase buffer length for smoother playback
                        maxMaxBufferLength: 120,        // Maximum buffer size during seeking
                        enableWorker: true,            // Use web workers for better performance
                        lowLatencyMode: false,         // Disable low latency for stability
                        backBufferLength: 60,          // Keep 60 seconds of past video in buffer
                        nudgeMaxRetry: 10,             // More retries when playback stalls
                        maxFragLookUpTolerance: 1.0,   // More tolerance for finding fragments
                        maxLoadingDelay: 4,            // Wait longer for segments to load
                        manifestLoadingMaxRetry: 6,    // More retries for manifest loading
                        levelLoadingMaxRetry: 6,       // More retries for playlist loading
                        fragLoadingMaxRetry: 6,        // More retries for fragment loading
                        startFragPrefetch: true,       // Start fetching next fragment early
                        testBandwidth: false           // Don't do bandwidth testing (more stable)
                    });
                    window.currentHls = hls;
                    // Add event listeners for better debugging
                    hls.on(Hls.Events.ERROR, function(event, data) {
                        console.error('HLS error:', data);
                        if (data.fatal) {
                            console.error('Fatal error:', data.type, data.details);
                            if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
                                console.log('Trying to recover from network error');
                                hls.startLoad();
                            } else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
                                console.log('Trying to recover from media error');
                                hls.recoverMediaError();
                            }
                        }
                    });
                    hls.on(Hls.Events.MANIFEST_PARSED, function() {
                        console.log('HLS manifest loaded successfully');
                        // Reset the video position to 0 before playing
                        video.currentTime = 0;
                        // Play after a slight delay to ensure proper initialization
                        setTimeout(() => {
                            video.currentTime = 0; // Reset again just to be sure
                            video.play().catch(err => console.log('Auto-play prevented:', err));
                        }, 100);
                    });
                    hls.on(Hls.Events.LEVEL_LOADED, function(event, data) {
                        console.log('HLS level loaded:', data);
                        // Set correct duration from the level details if available
                        if (data.details && data.details.totalduration) {
                            video.duration = data.details.totalduration;
                        }
                    });
                    hls.on(Hls.Events.FRAG_LOADING, function() { /* document.getElementById('loading-indicator').style.display = 'block'; */ });
                    hls.on(Hls.Events.FRAG_LOADED, function() { /* document.getElementById('loading-indicator').style.display = 'none'; */ });
                    // Load the source
                    video.currentTime = 0;
                    hls.loadSource(response.stream_url);
                    hls.attachMedia(video);
                } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                    // Native HLS support (Safari)
                    video.src = response.stream_url;
                    video.addEventListener('loadedmetadata', function() {
                        video.play();
                    });
                } else {
                    alert('Your browser does not support HLS streaming');
                }
            }
        });
    }
        
    // Display current search status
    socket.on('emit_show_search_status', (status) => {
      $('.image-search-status').html(status);
    });
  })

  //// RESPONDS TO SOCKET EVENTS
  //socket.on('emit_music_page_add_radio_state', (state) => {
  //  add_radio_state(state)
  //});
})();