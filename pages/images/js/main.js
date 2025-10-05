import FolderViewComponent from '/pages/FolderViewComponent.js';
import StarRatingComponent from '/pages/StarRating.js';
import PaginationComponent from '/pages/PaginationComponent.js';
import FileGridComponent from '/pages/FileGridComponent.js';


// Create a closed scope to avoid any variable collisions  
(function() {
  //// CONSTANTS AND VARIABLES
  let num_files_on_page = 60;
  let num_files_in_row = 6; // TODO: calculate from the screen size
  let selected_files = [];
  let currentImageMetadataFilePath = null; // To store currently opened image file path for metadata

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
  
  function renderImagePreview(fileData) { // Function for Images module
    const imageDataDiv = document.createElement('div');
    imageDataDiv.className = 'cell has-background-light p-1 is-flex is-flex-direction-column is-justify-content-space-between';

    const imageContainer = document.createElement('div');
    imageContainer.className = 'pswp-gallery__item is-flex-direction-column mb-2'; // Replicated from original code
    imageContainer.style.aspectRatio = '1'; // Replicated from original code
    imageContainer.style.display = 'flex'; // Replicated from original code
    imageContainer.style.justifyContent = 'center'; // Replicated from original code
    imageContainer.style.alignItems = 'center'; // Replicated from original code

    imageDataDiv.append(imageContainer);

    // Create a new star rating component (need to be created here to be accessible in the closure)
    const starRating = new StarRatingComponent({
      callback: (rating) => {
        console.log('New rating:', rating);
        socket.emit('emit_images_page_set_image_rating', {
          hash: fileData.hash,
          file_path: fileData.file_path,
          rating: rating,
        });
      },
      initialRating: fileData.file_info.user_rating,
    });
  
    // Create an object for holding the image data
    const image = document.createElement('img');
    image.src = 'image_files/'+fileData.file_path;
    image.onload = function() {
      // Create link to full image
      const link = document.createElement('a');
      // Align the image to the center of the a element
      link.style.display = 'flex';
      link.style.justifyContent = 'center';
      link.style.alignItems = 'center';
      link.style.width = '100%';
      link.style.height = '100%';
      // Set aspect ratio to maintain the square shape
      link.style.aspectRatio = '1';
  
      link.href = 'image_files/'+fileData.file_path;
      link.target = '_blank';
      link.setAttribute('data-pswp-width', image.width);
      link.setAttribute('data-pswp-height', image.height);
  
  
      // Reduce the size of the image to reduce the load on the page
      const reducedImage = resize_image(image, 400, 400);
      reducedImage.style.maxWidth = '100%'; // Ensure the image takes the maximum amount of space
      reducedImage.style.maxHeight = '100%'; // without exceeding the square's bounds
      reducedImage.style.objectFit = 'contain'; // Maintain the aspect ratio    
  
      // Append the reduced image to the link
      link.append(reducedImage);
  
      // Append the image to the beginning of div
      imageContainer.prepend(link);
  
  
      // Create a new div element
      const hiddenCaptionDiv = document.createElement('div');
      hiddenCaptionDiv.className = 'hidden-caption-content';
  
      // Set the innerHTML of the div to include the file path or any other HTML content
      let starRatingDiv = starRating.issueNewHtmlComponent({
        containerType: 'div',
        //size:3, 
        isActive: true
      });
      hiddenCaptionDiv.append(starRatingDiv);
      // Append the div to the imageContainer
      imageContainer.append(hiddenCaptionDiv);
  
      //$(imageContainer).append('<div class="hidden-caption-content">' + fileData.file_path + '</div>');
      image.remove();
    };

    return imageDataDiv;
  }

  function renderCustomData(fileData) { // Create renderCustomData function
    const dataContainer = document.createElement('div'); // Or any container element you used before

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
  
    const data = document.createElement('p'); // Or whatever element you used
    data.style.wordBreak = 'break-all'; // Keep styling if you had it
  
    $(data).append('<b>Path:</b>&nbsp;' + fileData.file_path + '<br>'); // **Adapt data access to fileData**
    $(data).append('<b>Hash:</b>&nbsp;' + fileData.hash + '<br>');
  
    // Create a new star rating component (need to be created here to be accessible in the closure)
    const starRating = new StarRatingComponent({
      initialRating: fileData.file_info.user_rating,
    });

    const StarRatingComponentObject = starRating.issueNewHtmlComponent({ // Use the *global* starRatingComponent
      containerType: 'span',
      //size:6,
      isActive: false
    })
    $(data).append('<br><b>User rating:</b>&nbsp;&nbsp;');
    $(data).append(StarRatingComponentObject)
    $(data).append('<br>');
  
  
    if (fileData.file_info.model_rating != null){
      $(data).append('<b>Model rating:</b>&nbsp;' + fileData.file_info.model_rating.toFixed(2) + '/10<br>');
    } else {
      $(data).append('<b>Model rating:</b>&nbsp;N/A<br>');
    }
  
    $(data).append('<b>File size:</b>&nbsp;' + fileData.file_size + '<br>');
    $(data).append('<b>Resolution:</b>&nbsp;' + fileData.file_info.resolution + '<br><br>');
  
    dataContainer.appendChild(data); // Append data to the container
  
    return dataContainer; // Return the container with all info
  }



  let lastActivatedCheckbox = null; // Create a variable to store the last activated checkbox

  // Add checkbox select method
  function selectCheckbox(currentCheckbox, isChecked) {
    currentCheckbox.checked = isChecked;
    const filePath = $(currentCheckbox).data('file-path');
    if (isChecked) {
      if (!selected_files.includes(filePath)) {
        selected_files.push(filePath);
      }
    } else {
      selected_files = selected_files.filter(function(value) {
        return value !== filePath;
      });
    }
  }

  function renderActions(fileData) { // Create renderActions function
    const levelContainer = document.createElement('div'); // Or any container element you used before
    levelContainer.className = 'level is-gapless mt-1'; // Add classes if you had them

    // Create level-left container
    const levelLeft = document.createElement('div');
    levelLeft.className = 'level-left is-gapless is-flex-direction-column';

    // Create level-right container
    const levelRight = document.createElement('div');
    levelRight.className = 'level-right is-gapless';

    // Add the level containers to the main level container
    levelContainer.append(levelLeft); // Add the level containers to the actions container
    levelContainer.append(levelRight); // Add the level containers to the actions container

    // Add two raws of buttons for levelLeft container to hold
    const btnsRow1 = document.createElement('div');
    btnsRow1.className = 'is-flex is-gapless is-justify-content-flex-start';
    btnsRow1.style.width = '100%';
    const btnsRow2 = document.createElement('div');
    btnsRow2.className = 'is-flex is-gapless is-justify-content-flex-start';
    btnsRow2.style.width = '100%';

    // Add the button rows to the levelLeft container
    levelLeft.append(btnsRow1);
    levelLeft.append(btnsRow2);

    // Create buttons for opening the file
    const btn_open = document.createElement('button');
    btn_open.className = 'button level-left is-gapless';
    btn_open.innerHTML = '<span class="icon"><i class="fas fa-folder-open"></i></span><span>Open</span>';
    btn_open.onclick = function() {
      console.log('Open file in folder: ' + fileData.full_path);
      socket.emit('emit_images_page_open_file_in_folder', fileData.full_path);
    };
    btnsRow1.append(btn_open);

    // Create a button for finding similar images
    const btn_find_similar = document.createElement('button');
    btn_find_similar.className = 'button level-left is-gapless';
    btn_find_similar.innerHTML = '<span class="icon"><i class="fas fa-search"></i></span><span>Find similar</span>';
    btn_find_similar.onclick = function() {
      console.log('Find similar images for: ' + fileData.full_path);

      let url = new URL(window.location.href);
      let params = new URLSearchParams(url.search);
      params.set('text_query', fileData.full_path);
      params.set('page', 1);
      url.search = params.toString();
      window.location.href = url.toString();
    };
    btnsRow1.append(btn_find_similar);

    // --- Add the Edit Internal Metadata button (temporarily disabled) ---
    const btn_edit_internal_metadata = document.createElement('button');
    btn_edit_internal_metadata.disabled = true; // Temporarily disable this button
    btn_edit_internal_metadata.className = 'button level-left is-gapless';
    btn_edit_internal_metadata.innerHTML = '<span class="icon"><i class="fas fa-file"></i></span><span>Edit meta</span>';
    btn_edit_internal_metadata.onclick = function() {
        // console.log('Edit internal metadata for: ' + fileData.file_path);
        // currentImageMetadataFilePath = fileData.file_path; // Set the global tracker
        // $('#metadata_modal_title').text(`Edit ${fileData.base_name}.meta`); // Set modal title
        // $('#metadata_content_textarea').val('Loading metadata...'); // Show loading state
        // socket.emit('emit_images_page_get_image_metadata_file_content', fileData.file_path);
        // $('#metadata_editor_modal').addClass('is-active'); // Show modal
    };
    btnsRow2.append(btn_edit_internal_metadata);

    // --- Add the Edit External Metadata button ---
    const btn_edit_external_metadata = document.createElement('button');
    btn_edit_external_metadata.className = 'button level-left is-gapless';
    btn_edit_external_metadata.innerHTML = '<span class="icon"><i class="fas fa-question"></i></span><span>Edit .meta</span>'; // Changed to 'Edit Meta' for space
    btn_edit_external_metadata.onclick = function() {
        console.log('Edit metadata for: ' + fileData.file_path);
        currentImageMetadataFilePath = fileData.file_path; // Set the global tracker
        $('#metadata_modal_title').text(`Edit ${fileData.base_name}.meta`); // Set modal title
        $('#metadata_content_textarea').val('Loading metadata...'); // Show loading state
        socket.emit('emit_images_page_get_image_metadata_file_content', fileData.file_path);
        $('#metadata_editor_modal').addClass('is-active'); // Show modal
    };
    btnsRow2.append(btn_edit_external_metadata);

    // Create a checkbox for selecting the file for further actions
    const checkboxLabel = document.createElement('label');
    checkboxLabel.className = 'b-checkbox checkbox is-large level-right mr-0 ';
    checkboxLabel.innerHTML = /*html*/`<input type="checkbox" value="false">
                          <span class="check is-success"></span>`;
    const checkboxLabelInput = checkboxLabel.querySelector('input');
    checkboxLabelInput.dataset.filePath = fileData.full_path;

    // Handle the checkbox click event
    checkboxLabelInput.onclick = function(event) {
      //event.stopPropagation();
      const isShiftPressed = event.shiftKey;
      const checkboxes = $("input[type='checkbox']");
      const isChecked = this.checked;

      if (isShiftPressed) {
        console.log('Shift is pressed');
        
        if (!isChecked) lastActivatedCheckbox = null;
    
        if (lastActivatedCheckbox === null) {
          // No checkbox was activated before, select all checkboxes
          checkboxes.each(function() {
            selectCheckbox(this, isChecked);
          });
        } else {
          // Select all checkboxes from last activated to current one
          let start = checkboxes.index(lastActivatedCheckbox);
          let end = checkboxes.index(this);
          console.log('start', start, 'end', end);
          if (start > end) [start, end] = [end, start]; // Ensure start is less than end

          checkboxes.slice(start, end + 1).each(function() {
            selectCheckbox(this, isChecked);
          });
        }
      } else {
        console.log('Shift is not pressed');
        lastActivatedCheckbox = this.checked ? this : null;

        // Check if the checkbox is activated
        selectCheckbox(this, isChecked);
      }

      // Update the counter of selected files
      let selectedCount = $("input[type='checkbox']:checked").length;
      let msg = selectedCount + " file" + (selectedCount !== 1 ? "s" : "") + " selected";
      $("#selected_files_counter").text(msg);

      // Show files_actions window if there are selected files or hide it if there none
      if ($("input[type='checkbox']:checked").length > 0){
        $('#files_actions').show();
      } else {
        $('#files_actions').hide();
        selected_files = [];
      }

      console.log('selected_files', selected_files);
    };
    levelRight.append(checkboxLabel);

    return levelContainer; // Return the container with all info
  }

  function resize_image(image, maxWidth, maxHeight) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set the dimensions of the canvas to the desired dimensions of the image
    let width = image.width;
    let height = image.height;

    // Calculate the width and height, maintaining the aspect ratio
    if (width > height) {
      if (width > maxWidth) {
        height *= maxWidth / width;
        width = maxWidth;
      }
    } else {
      if (height > maxHeight) {
        width *= maxHeight / height;
        height = maxHeight;
      }
    }

    // Set the canvas width and height and draw the image data into the canvas
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(image, 0, 0, width, height);

    // Get the reduced image data from the canvas
    const reducedImage = new Image();
    reducedImage.src = canvas.toDataURL();

    return reducedImage;
  }

  function isSubfolder(parentPath, childPath) {
    // Normalize paths to remove any '..' or '.' segments
    const normalizedParentPath = new URL(parentPath, 'file://').pathname.replace(/\/$/, '') + '/';
    const normalizedChildPath = new URL(childPath, 'file://').pathname.replace(/\/$/, '') + '/';
  
    // Check if the child path starts with the parent path
    return normalizedChildPath.startsWith(normalizedParentPath);
  }

  function create_folder_representation(folders_dict, active_path = '', current_path = '') {
    let folderRepresentation = '';

    const folderName = folders_dict.name;
    const numImages = folders_dict.num_images;
    const totalImages = folders_dict.total_images;
    let current_path_ = current_path + folderName + '/';
    const isActive = active_path === current_path_ ? 'is-active' : '';
    let encoded_link = `path=${encodeURIComponent(current_path_)}`;
    let color = isSubfolder(active_path, current_path_) ? 'has-text-black' : 'has-text-grey-light';
    if (active_path === current_path_) color = '';

    // Conditional check for displaying image counts
    let imageCountDisplay = numImages === totalImages ? `[${numImages}]` : `[${numImages} | ${totalImages}]`;
    folderRepresentation += /*html*/`<li><a class="${isActive} ${color}" href="?${encoded_link}">${folderName} ${imageCountDisplay}</a>`;


    // Sort the folders by name
    const sortedFolders = Object.keys(folders_dict.subfolders).sort((a, b) => {
      return folders_dict.subfolders[a].name.localeCompare(folders_dict.subfolders[b].name);
    });

    folderRepresentation += '<ul>';

    // Create a new folder representation for each subfolder
    for (const folderKey of sortedFolders) {
      const folder = folders_dict.subfolders[folderKey];
      let current_path_ = current_path + folderName + '/';

      if (isSubfolder(current_path_, active_path)) {
        folderRepresentation += create_folder_representation(folder, active_path, current_path_);
      }
    } 

    folderRepresentation += '</ul>';
    folderRepresentation += '</li>';

    return folderRepresentation;
  }

  //// AFTER PAGE LOADED
  $(document).ready(function() {
    let paginationComponent;

    // Request current media folder path
    socket.emit('emit_images_page_get_path_to_media_folder');

    // Request files from the main media folder
    // --- Folder View ---
    socket.emit('emit_images_page_get_folders', {
      path: path, 
    }, (response) => { // event handler for folders
      console.log('emit_images_page_get_folders response', response);

      const folderView = new FolderViewComponent(response.folders, response.folder_path, false); // Enable context menu
      $('.menu').append(folderView.getDOMElement());
    });
    
    // --- File List ---
    socket.emit('emit_images_page_get_files', {
      path: path, 
      pagination: (page-1)*num_files_on_page, 
      limit: page * num_files_on_page,
      text_query: text_query,
      seed: seed
    }, (response)=>{
      console.log('emit_images_page_get_files response', response);
      // Handle the response here
      //all_files_paths = response["all_files_paths"];

      // Update or Initialize FileGridComponent
      const fileGridComponent = new FileGridComponent({
          containerId: '#images_grid_container', // Use the new container ID
          filesData: response.files_data,
          renderPreviewContent: renderImagePreview, // Pass the text preview function
          renderCustomData: renderCustomData, // Pass the custom data rendering function
          renderActions: renderActions, // Pass the actions rendering function
          handleFileClick: (fileData) => {
            // ?
          },
          numColumns: num_files_in_row, 
          minTileWidth: "18rem",
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



      // Initialize photoSwipe gallery
      window.photoGalleryLightbox.on('uiRegister', function() {
        window.photoGalleryLightbox.pswp.ui.registerElement({
          name: 'custom-caption',
          order: 9,
          isButton: false,
          appendTo: 'root',
          html: 'Caption text',
          onInit: (el, pswp) => {
            console.log('init', el, pswp);
            let hiddenCaption = null;
            let hiddenCaptionOriginalParent = null;

            window.photoGalleryLightbox.pswp.on('change', () => {
              const currSlideElement = window.photoGalleryLightbox.pswp.currSlide.data.element;

              if (hiddenCaptionOriginalParent) {
                hiddenCaptionOriginalParent.append(hiddenCaption);
                hiddenCaption.style.display = 'none';
              }

              if (currSlideElement) {
                hiddenCaption = currSlideElement.querySelector('.hidden-caption-content');

                if (hiddenCaption) {
                  hiddenCaptionOriginalParent = hiddenCaption.parentElement;

                  // get caption from element with class hidden-caption-content
                  //captionHTML = hiddenCaption.innerHTML;
                  el.innerHTML = '';
                  hiddenCaption.style.display = 'block';
                  el.append(hiddenCaption);
                } else {
                  // get caption from alt attribute
                  //captionHTML = currSlideElement.querySelector('img').getAttribute('alt');
                }
              }
              //el.innerHTML = captionHTML || '';
            });

            window.photoGalleryLightbox.pswp.on('destroy', () => {
              if (hiddenCaptionOriginalParent) {
                hiddenCaptionOriginalParent.append(hiddenCaption);
                hiddenCaption.style.display = 'none';
              }
            });
          }
        });
      });
      window.photoGalleryLightbox.init();
    });


    // Display current search status
    socket.on('emit_show_search_status', (status) => {
      $('.image-search-status').html(status);
    });

    // Show current media folder path
    socket.on('emit_images_page_show_path_to_media_folder', (current_path) => {
      $('#path_to_media_folder').val(current_path);
    });

    socket.on('emit_images_page_show_image_metadata_content', (data) => {
      const file_path_received = data.file_path;
      const metadata_content = data.content;
      
      // Only update if the received content is for the currently active modal
      if (file_path_received === currentImageMetadataFilePath) {
        $('#metadata_content_textarea').val(metadata_content);
      }
    });

    // Set search query in input 
    $('#search_input').val(text_query);

    // Search for images
    $('#seach_button').click(function() {
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
      $('#seach_button').click();
    });

    // Update path to the media folder
    $(`#update_path_to_media_folder`).click(()=>{ 
      socket.emit('emit_images_page_update_path_to_media_folder', $('#path_to_media_folder').val());
      // refresh page after a second
      setTimeout(function(){
        location.reload();
      }, 500);
    });

    // Unselect all files
    $('#unselect_all_files').click(function() {
      $("input[type='checkbox']").prop('checked', false);
      $('#files_actions').hide();
    });

    // Delete selected files
    $('#delete_selected_files').click(function() {
      //console.log('Delete selected files: ' + selected_files);
      socket.emit('emit_images_page_send_files_to_trash', selected_files);
      // refresh page
      location.reload();
    });

    // Move selected files to the new folder
    $('#move_selected_files').click(function() {
      // Activate the modal window
      $('#move_files_modal').addClass('is-active');
    });

    // Close the modal window
    $('#move_files_modal .modal-close-action').click(function() {
      $('#move_files_modal').removeClass('is-active');
    });

    // Move selected files to the new folder if confirmed
    $('#move_files_modal .modal-confirm-action').click(function() {
      let target_folder = $('#move_files_modal input').val();
      socket.emit('emit_images_page_move_files', {
        files: selected_files,
        target_folder: target_folder
      });
      // refresh page
      location.reload();
    });

    // Close modal when clicking modal-close elements (background, X button, Cancel button)
    $('#metadata_editor_modal .modal-close-action').on('click', function() {
        $('#metadata_editor_modal').removeClass('is-active'); // Hide modal
        currentImageMetadataFilePath = null; // Clear the tracker
    });

    // Save button in modal
    $('#metadata_editor_modal .modal-save-action').on('click', function() {
        if (currentImageMetadataFilePath) { // Ensure a file is selected
            const metadataContent = $('#metadata_content_textarea').val(); // Get content from textarea
            socket.emit('emit_images_page_save_image_metadata_file_content', { 
                file_path: currentImageMetadataFilePath, 
                metadata_content: metadataContent 
            }); 
            $('#metadata_editor_modal').removeClass('is-active'); // Hide modal after save
            currentImageMetadataFilePath = null; // Clear the tracker
        }
    });
  })

  //// RESPONDS TO SOCKET EVENTS
  //socket.on('emit_music_page_add_radio_state', (state) => {
  //  add_radio_state(state)
  //});
})();