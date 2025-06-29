import FolderViewComponent from '/pages/FolderViewComponent.js';
import FileGridComponent from '/pages/FileGridComponent.js';

// Create a closed scope to avoid any variable collisions  
(function() {
  //// CONSTANTS AND VARIABLES
  let num_images_on_page = 20;

  //// BEFORE PAGE LOADED
  // read page number from the URL ?page=1
  const urlParams = new URLSearchParams(window.location.search);

  let pageParam = parseInt(urlParams.get('page'));
  let page = (!pageParam || pageParam < 1) ? 1 : pageParam;
  let text_query = urlParams.get('text_query') || '';

  let path = urlParams.get('path') || '';
  text_query = decodeURIComponent(text_query);
  path = decodeURIComponent(path);
  console.log('path', path);

  let seed = urlParams.get('seed');
  if (!seed) {
    seed = Math.floor(Math.random() * 1e9);
    urlParams.set('seed', seed);
    window.location.search = urlParams.toString();
  }
  

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
    image.src = '/video_files/' + fileData.preview_path; // Use preview_path
    image.style.maxWidth = '100%';
    image.style.maxHeight = '100%';
    image.style.objectFit = 'contain';
    videoPreviewContainer.appendChild(image);
    return videoPreviewContainer;
  }

  // function isSubfolder(parentPath, childPath) {
  //   // Normalize paths to remove any '..' or '.' segments
  //   const normalizedParentPath = new URL(parentPath, 'file://').pathname.replace(/\/$/, '') + '/';
  //   const normalizedChildPath = new URL(childPath, 'file://').pathname.replace(/\/$/, '') + '/';
  
  //   // Check if the child path starts with the parent path
  //   return normalizedChildPath.startsWith(normalizedParentPath);
  // }

  // function create_folder_representation(folders_dict, active_path = '', current_path = '') {
  //   let folderRepresentation = '';
    
  //   for (const folder in folders_dict) {
  //     console.log('folder', folder);
  //     let folder_dict = folders_dict[folder];
  //     console.log('folder_dict', folder_dict);
  //     let current_path_ = current_path + folder + '/';
  //     console.log('current_path', current_path_);
  //     const isActive = active_path === current_path_ ? 'is-active' : '';
  //     let encoded_link = `path=${encodeURIComponent(current_path_)}`;
  //     folderRepresentation += `<li><a class="${isActive}" href="?${encoded_link}">${folder}</a>`;
  //     console.log('!!!', active_path, current_path_);
  //     if (isSubfolder(current_path_, active_path)) {
  //       folderRepresentation += '<ul>';
  //       folderRepresentation += create_folder_representation(folder_dict, active_path, current_path_);
  //       folderRepresentation += '</ul>';
  //     }
  //     folderRepresentation += '</li>';
  //   }
  //   return folderRepresentation;
  // }

  function renderCustomData(fileData) {
    const dataContainer = document.createElement('div');
    dataContainer.className = 'file-custom-data';
    dataContainer.style.wordBreak = 'break-word';
    
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
    userRatingElement.innerHTML = `<b>User rating:</b>&nbsp;${fileData.user_rating !== null ? fileData.user_rating.toFixed(1) : 'N/A'}`;
    dataContainer.appendChild(userRatingElement);

    // Model Rating
    const modelRatingElement = document.createElement('p');
    modelRatingElement.className = 'file-info file-model-rating';
    modelRatingElement.innerHTML = `<b>Model rating:</b>&nbsp;${fileData.model_rating !== null ? fileData.model_rating.toFixed(1) : 'N/A'}`;
    dataContainer.appendChild(modelRatingElement);

    // File Size
    const fileSizeElement = document.createElement('p');
    fileSizeElement.className = 'file-info file-size';
    fileSizeElement.innerHTML = `<b>Size:</b>&nbsp;${fileData.file_size}`;
    dataContainer.appendChild(fileSizeElement);

    // Length
    const lengthElement = document.createElement('p');
    lengthElement.className = 'file-info file-length';
    lengthElement.innerHTML = `<b>Length:</b>&nbsp;${fileData.length}`;
    dataContainer.appendChild(lengthElement);
    
    // Last Played
    const lastPlayedElement = document.createElement('p');
    lastPlayedElement.className = 'file-info file-last-played';
    lastPlayedElement.innerHTML = `<b>Last Played:</b>&nbsp;${fileData.last_played}`;
    dataContainer.appendChild(lastPlayedElement);

    return dataContainer;
  }

  function create_folder_representation(folders_dict, active_path = '') {
    const folderView = new FolderViewComponent(folders_dict, active_path);
    return folderView.getDOMElement().innerHTML; // Return innerHTML to be compatible with old append logic
  }

  //// AFTER PAGE LOADED
  $(document).ready(function() {
    const modal = document.getElementById('video-modal');
    const openBtn = document.getElementById('open-video-modal');
    const closeBtn = modal.querySelector('.modal-close');
    const modalBg = modal.querySelector('.modal-background');
    const video = document.getElementById('modal-video-player');

    // Request files from the main media folder
    socket.emit('emit_videos_page_get_files', {
      path: path, 
      pagination: (page-1)*num_images_on_page, 
      limit: page * num_images_on_page,
      text_query: text_query,
      seed: seed
    });

    // Display files from the folder
    socket.on('emit_videos_page_show_files', (data) => {
      console.log('emit_videos_page_show_files', data);


      // Instantiate FileGridComponent
      const fileGridComponent = new FileGridComponent({
        containerId: '#images_preview_container', // Use the new container ID
        filesData: data.files_data,
        renderPreviewContent: renderVideoPreview, // Pass the video preview function
        renderCustomData: renderCustomData,      // Pass the custom data rendering function
        handleFileClick: (fileData) => {
            // Logic to open modal and play video
            modal.classList.add('is-active');
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
        },
        numColumns: 5, // As in original code
      });

      // update the pagination
      $(".pagination-list").empty();
      let total_pages = Math.ceil(data["total_files"] / num_images_on_page);

      for (let i = 1; i <= total_pages; i++) {
        // only include the first, one before, one after, and last pages
        if (i == 1 || i == page - 1 || i == page || i == page + 1 || i == total_pages) {
          //let link = `page=${i}`;
          urlParams.set('page', i);
          
          if (text_query){
            let encoded_text_query = encodeURIComponent(text_query);
            //link = link + `&text_query=${encoded_text_query}`
            urlParams.set('text_query', encoded_text_query);
          }
          console.log('urlParams', urlParams.toString());

          let template = `<li>
            <a href="?${urlParams.toString()}" class="pagination-link ${i == page?'is-current':''}" aria-label="Goto page ${i}">${i}</a>
          </li>`
          $(".pagination-list").append(template);
        }
        // add ellipsis when there are skipped pages
        else if (i == 2 && page > 3 || i == total_pages - 1 && page < total_pages - 2) {
          let template = `<li>
            <span class="pagination-ellipsis">&hellip;</span>
          </li>`
          $(".pagination-list").append(template);
        }
      }

      // Create folder representation from data["folders"] dictionary
      const folderViewComponent = new FolderViewComponent(data["folders"], data["folder_path"]);
      // Add the folder representation to the page
      $('#folders_menu').empty().append(folderViewComponent.getDOMElement());
      
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

    // Display current search status
    socket.on('emit_videos_page_show_search_status', (status) => {
      $('.image-search-status').html(status);
    });


    // Set search query in input 
    $('#search_input').val(text_query);

    // Search for images
    $('#search_button').click(function() {
      let text_query = $('#search_input').val();
      let url = new URL(window.location.href);
      let params = new URLSearchParams(url.search);
      params.set('text_query', text_query);
      params.set('page', 1);
      // Generate a new seed for each search
      let newSeed = Math.floor(Math.random() * 1e9);
      params.set('seed', newSeed);
      url.search = params.toString();
      window.location.href = url.toString();
    });

    $('.set_search').click(function() {
      $('#search_input').val($(this).text());
      $('#search_button').click();
    });
  })

  //// RESPONDS TO SOCKET EVENTS
  //socket.on('emit_music_page_add_radio_state', (state) => {
  //  add_radio_state(state)
  //});
})();