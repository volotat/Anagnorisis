import FolderViewComponent from '/pages/FolderViewComponent.js';
import StarRatingComponent from '/pages/StarRating.js';
import PlaylistManager from '/pages/music/js/PlaylistManager.js';
import SongControlPanel from '/pages/music/js/SongControlPanel.js';
import PaginationComponent from '/pages/PaginationComponent.js';
import FileGridComponent from '/pages/FileGridComponent.js';


//// BEFORE PAGE LOADED

function renderImagePreview(fileData) { // Function for Images module preview
    const imageDataDiv = document.createElement('div');
    imageDataDiv.className = 'cell has-background-light p-1 is-flex is-flex-direction-column is-justify-content-space-between';

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
    if (fileData.search_total_score > 0) {
        const searchScoresElement = document.createElement('p');
        searchScoresElement.className = 'file-info file-search-scores';
        const searchScores = [
            (fileData.search_total_score || 0).toFixed(3),
            (fileData.search_semantic_score || 0).toFixed(3),
            (fileData.search_meta_score || 0).toFixed(3),
        ];
        searchScoresElement.innerHTML = `<b>Search Scores:</b>&nbsp;${searchScores.join('/')}`;
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
  let num_files_on_page = 30;
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

  //window.global_function = (va1, var2){
  //  
  //}


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
      limit: page * num_files_on_page,
      text_query: text_query,
      seed: seed
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
          handleFileClick: (fileData) => {
            // ?
          },
          numColumns: num_files_in_row, 
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

      // Activate "Update Playlist" buttons
      $('.update_playlist').prop('disabled', false);

      // // Create a container for the images
      // let container = /*html*/`<div class="fixed-grid has-${num_files_in_row}-cols is-gap-0.5">
      //   <div class="grid" id="images_grid_container">
      //   </div>
      // </div>`;
      // // Add the container to the body (or another container)
      // $('#images_preview_container').append(container);

      // // Create a checkbox tracking variables
      // let lastActivatedCheckbox = null;

      // // Add checkbox select method
      // function selectCheckbox(currentCheckbox, isChecked) {
      //   currentCheckbox.checked = isChecked;
      //   const filePath = $(currentCheckbox).data('rel-path');
      //   if (isChecked) {
      //     if (!selected_files.includes(filePath)) {
      //       selected_files.push(filePath);
      //     }
      //   } else {
      //     selected_files = selected_files.filter(function(value) {
      //       return value !== filePath;
      //     });
      //   }
      // }

      // // Create a new div for each image
      // data["files_data"].forEach(item => {
      //   const imageDataDiv = document.createElement('div');
      //   imageDataDiv.className = 'cell has-background-light p-1 is-flex is-flex-direction-column is-justify-content-space-between';

      //   const imageContainer = document.createElement('div');
      //   imageContainer.classList.add('pswp-gallery__item', 'is-flex-direction-column');
      //   // Make image container always square
      //   imageContainer.style.aspectRatio = 1;
      //   // Horizontally center the image
      //   imageContainer.style.display = 'flex';
      //   imageContainer.style.justifyContent = 'center'; 
      //   imageContainer.style.alignItems = 'center';
      //   $(imageContainer).addClass('mb-2');


      //   imageDataDiv.append(imageContainer);

      //   // Create a new star rating component
      //   const callback = (rating) => {
      //     console.log('New rating:', rating);
      //     /*socket.emit('emit_music_page_set_music_rating', {
      //       hash: item.hash,
      //       file_path: item.file_path,
      //       rating: rating,
      //     });*/
      //   }
      //   const starRating = new StarRatingComponent({
      //     callback: callback,
      //     initialRating: item.user_rating, //parseInt(Math.random() * 11),
      //   });

      //   // Create an object for holding the image data
      //   if (item.audiofile_data['image'] == null){
      //     // Create an icon with music note from font awesome
      //     const icon = document.createElement('span');
      //     icon.className = 'icon is-large is-fullwidth';
      //     icon.innerHTML = '<i class="fas fa-6x fa-music"></i>';
      //     imageContainer.append(icon);
      //   } else {
      //     const image = document.createElement('img');
      //     image.src = item.audiofile_data['image']; 
      //     imageContainer.append(image);
      //   }
      //   //'music_files/'+item.file_path;
      //   /*
      //   image.onload = function() {
      //     // Create link to full image
      //     const link = document.createElement('a');
      //     // Align the image to the center of the a element
      //     link.style.display = 'flex';
      //     link.style.justifyContent = 'center';
      //     link.style.alignItems = 'center';
      //     link.style.width = '100%';
      //     link.style.height = '100%';
      //     // Set aspect ratio to maintain the square shape
      //     link.style.aspectRatio = '1';
          
      //     link.href = 'music_files/'+item.file_path;
      //     link.target = '_blank';
      //     link.setAttribute('data-pswp-width', image.width);
      //     link.setAttribute('data-pswp-height', image.height);

      //     // Reduce the size of the image to reduce the load on the page
      //     const reducedImage = resize_image(image, 400, 400);
      //     reducedImage.style.maxWidth = '100%'; // Ensure the image takes the maximum amount of space
      //     reducedImage.style.maxHeight = '100%'; // without exceeding the square's bounds
      //     reducedImage.style.objectFit = 'contain'; // Maintain the aspect ratio    
          
      //     // Append the reduced image to the link
      //     link.append(reducedImage);

      //     // Append the image to the beginning of div
      //     imageContainer.prepend(link);

          
      //     // Create a new div element
      //     const hiddenCaptionDiv = document.createElement('div');
      //     hiddenCaptionDiv.className = 'hidden-caption-content';

      //     // Set the innerHTML of the div to include the file path or any other HTML content
      //     let starRatingDiv = starRating.issueNewHtmlComponent({
      //       containerType: 'div',
      //       size:3, 
      //       isActive: true
      //     });
      //     hiddenCaptionDiv.append(starRatingDiv);
      //     // Append the div to the imageContainer
      //     imageContainer.append(hiddenCaptionDiv);
          

      //     //$(imageContainer).append('<div class="hidden-caption-content">' + item.file_path + '</div>');
      //     image.remove();
      //   };*/

      //   // Add name of the file to the div
      //   const data = document.createElement('p');
      //   data.style.wordBreak = 'break-all'; // break long words

      //   $(data).append('<b>Path:</b>&nbsp;' + item.file_path + '<br>');
      //   $(data).append('<b>Hash:</b>&nbsp;' + item.hash + '<br>');
        

      //   const StarRatingComponentObject = starRating.issueNewHtmlComponent({
      //     containerType: 'span',
      //     //size:6, 
      //     isActive: false
      //   })
      //   $(data).append('<br><b>User rating:</b>&nbsp;&nbsp;');
      //   $(data).append(StarRatingComponentObject)
      //   $(data).append('<br>');

      //   if (item.model_rating != null){
      //     $(data).append('<b>Model rating:</b>&nbsp;' + item.model_rating.toFixed(2) + '/10<br>');
      //   } else {
      //     $(data).append('<b>Model rating:</b>&nbsp;N/A<br>');
      //   }
        
      //   $(data).append('<b>File size:</b>&nbsp;' + item.file_size + '<br>');
      //   $(data).append('<b>Length:</b>&nbsp;' + item.length + '<br>');
      //   $(data).append('<b>Last played:</b>&nbsp;' + item.last_played + '<br><br>');

      //   imageDataDiv.append(data);

        
      //   /*data.innerHTML = '<b>Path:</b> ' + item.file_path;
      //   data.innerHTML += '<br><b>Hash:</b> ' + item.hash;
      //   data.innerHTML += '<br><b>User rating:</b> ' + starRating.issueNewHtmlComponent({
      //     size:1, 
      //     isActive: false
      //   }).innerHTML; //+ item.user_rating;
      //   data.innerHTML += '<br><b>Model rating:</b> ' + item.model_rating;
      //   data.innerHTML += '<br><b>File size:</b> ' + item.file_size;
      //   data.innerHTML += '<br><b>Resolution:</b> ' + item.resolution;
      //   data.innerHTML += '<br><br>';*/

      //   // Create a level container
      //   const levelContainer = document.createElement('div');
      //   levelContainer.className = 'level is-gapless';
      //   imageDataDiv.append(levelContainer);

      //   // Create level-left container
      //   const levelLeft = document.createElement('div');
      //   levelLeft.className = 'level-left is-gapless';

      //   // Create level-right container
      //   const levelRight = document.createElement('div');
      //   levelRight.className = 'level-right is-gapless';

      //   // Add the level containers to the level container
      //   levelContainer.append(levelLeft);
      //   levelContainer.append(levelRight);

      //   // Create buttons for opening the file
      //   const btn_open = document.createElement('button');
      //   btn_open.className = 'button level-left is-gapless';
      //   btn_open.innerHTML = '<span class="icon"><i class="fas fa-folder-open"></i></span><span>Open</span>';
      //   btn_open.onclick = function() {
      //     console.log('Open file in folder: ' + item.full_path);
      //     socket.emit('emit_music_page_open_file_in_folder', item.full_path);
      //   };
      //   levelLeft.append(btn_open);

      //   // Create a button for finding similar music files
      //   const btn_find_similar = document.createElement('button');
      //   btn_find_similar.className = 'button level-left is-gapless';
      //   btn_find_similar.innerHTML = '<span class="icon"><i class="fas fa-search"></i></span><span>Find similar</span>';
      //   btn_find_similar.onclick = function() {
      //     console.log('Find similar images for: ' + item.full_path);

      //     let url = new URL(window.location.href);
      //     let params = new URLSearchParams(url.search);
      //     params.set('text_query', item.full_path);
      //     params.set('page', 1);
      //     url.search = params.toString();
      //     window.location.href = url.toString();
      //   };
      //   levelLeft.append(btn_find_similar);

      //   // Create a checkbox for selecting the file for further actions
      //   const checkboxLabel = document.createElement('label');
      //   checkboxLabel.className = 'b-checkbox checkbox is-large level-right mr-0 ';
      //   checkboxLabel.innerHTML = /*html*/`<input type="checkbox" value="false">
      //                         <span class="check is-success"></span>`;
      //   const checkboxLabelInput = checkboxLabel.querySelector('input');
      //   checkboxLabelInput.dataset.filePath = item.full_path;
      //   checkboxLabelInput.dataset.relPath = item.file_path;

      //   // Handle the checkbox click event
      //   checkboxLabelInput.onclick = function(event) {
      //     //event.stopPropagation();
      //     const isShiftPressed = event.shiftKey;
      //     const checkboxes = $("input[type='checkbox']");
      //     const isChecked = this.checked;

      //     if (isShiftPressed) {
      //       console.log('Shift is pressed');
            
      //       if (!isChecked) lastActivatedCheckbox = null;
        
      //       if (lastActivatedCheckbox === null) {
      //         // No checkbox was activated before, select all checkboxes
      //         checkboxes.each(function() {
      //           selectCheckbox(this, isChecked);
      //         });
      //       } else {
      //         // Select all checkboxes from last activated to current one
      //         let start = checkboxes.index(lastActivatedCheckbox);
      //         let end = checkboxes.index(this);
      //         console.log('start', start, 'end', end);
      //         if (start > end) [start, end] = [end, start]; // Ensure start is less than end

      //         checkboxes.slice(start, end + 1).each(function() {
      //           selectCheckbox(this, isChecked);
      //         });
      //       }
      //     } else {
      //       console.log('Shift is not pressed');
      //       lastActivatedCheckbox = this.checked ? this : null;

      //       // Check if the checkbox is activated
      //       selectCheckbox(this, isChecked);
      //     }

      //     // Update the counter of selected files
      //     let selectedCount = $("input[type='checkbox']:checked").length;
      //     let msg = selectedCount + " file" + (selectedCount !== 1 ? "s" : "") + " selected";
      //     $("#selected_files_counter").text(msg);

      //     // Show files_actions window if there are selected files or hide it if there none
      //     if ($("input[type='checkbox']:checked").length > 0){
      //       $('#files_actions').show();
      //     } else {
      //       $('#files_actions').hide();
      //       selected_files = [];
      //     }

      //     console.log('selected_files', selected_files);
      //   };
      //   levelRight.append(checkboxLabel);


      //   /*
      //   // Create a button for deleting the file
      //   btn_delete = document.createElement('button');
      //   btn_delete.className = 'button is-pulled-right';
      //   btn_delete.innerHTML = '<span class="icon"><i class="fas fa-trash"></i></span><span>Delete</span>';
      //   btn_delete.onclick = function() {
      //     console.log('Delete file: ' + item.full_path)
      //     socket.emit('emit_music_page_send_file_to_trash', item.full_path);
      //     // refresh page
      //     location.reload();
      //   };
      //   data.append(btn_delete);
      //   */

      //   //name.className = 'has-text-centered';
        

      //   $('#images_grid_container').append(imageDataDiv);
      //   //window.photoGalleryLightbox.init();
      // });

      // // Update or Initialize Pagination Component
      // const paginationContainer = $('.pagination.is-rounded.level-left.mb-0 .pagination-list'); // Select pagination container
      // const urlParams = new URLSearchParams(window.location.search); // Get URL parameters for pattern
      // // let urlPattern = `?page={page}`; // Base URL pattern

      // // if (urlParams.get('text_query')) { // Add text_query if present
      // //     urlPattern += `&text_query=${encodeURIComponent(urlParams.get('text_query'))}`;
      // // }

      // let urlPattern = `?${urlParams.toString()}`;

      // if (!paginationComponent) { // Instantiate PaginationComponent if it doesn't exist yet
      //     paginationComponent = new PaginationComponent({
      //         containerId: paginationContainer.closest('.pagination').get(0), // Pass the pagination nav element
      //         currentPage: page,
      //         totalPages: Math.ceil(data["total_files"] / num_files_on_page),
      //         urlPattern: urlPattern,
      //     });
      // } else { // Update existing PaginationComponent
      //     paginationComponent.updatePagination(page, Math.ceil(data["total_files"] / num_images_on_page));
      // }

      // // Create folder representation from data["folders"] dictionary
      // const folderView = new FolderViewComponent(data["folders"], data["folder_path"]);

      // // Add the folder representation to the page
      // $('.menu').append(folderView.getDOMElement());
    });

    // Display current search status
    socket.on('emit_show_search_status', (status) => {
      $('.music-search-status').html(status);
    });

    // Show current media folder path
    socket.on('emit_music_page_show_path_to_media_folder', (current_path) => {
      $('#path_to_media_folder').val(current_path);
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

    // Set search query in input
    $('.set_search').click(function() {
      $('#search_input').val($(this).text());
      $('#search_button').click();
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
  })

  //// RESPONDS TO SOCKET EVENTS
  //socket.on('emit_music_page_add_radio_state', (state) => {
  //  add_radio_state(state)
  //});
})();