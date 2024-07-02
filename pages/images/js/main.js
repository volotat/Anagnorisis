// Create a closed scope to avoid any variable collisions  
(function() {
  //// CONSTANTS AND VARIABLES
  let num_images_on_page = 30;

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
    
    for (const folder in folders_dict) {
      console.log('folder', folder);
      let folder_dict = folders_dict[folder];
      console.log('folder_dict', folder_dict);
      let current_path_ = current_path + folder + '/';
      console.log('current_path', current_path_);
      const isActive = active_path === current_path_ ? 'is-active' : '';
      let encoded_link = `path=${encodeURIComponent(current_path_)}`;
      folderRepresentation += `<li><a class="${isActive}" href="?${encoded_link}">${folder}</a>`;
      console.log('!!!', active_path, current_path_);
      if (isSubfolder(current_path_, active_path)) {
        folderRepresentation += '<ul>';
        folderRepresentation += create_folder_representation(folder_dict, active_path, current_path_);
        folderRepresentation += '</ul>';
      }
      folderRepresentation += '</li>';
    }
    return folderRepresentation;
  }

  //// AFTER PAGE LOADED
  $(document).ready(function() {
    // Request files from the main media folder
    socket.emit('emit_images_page_get_files', {
      path: path, 
      pagination: (page-1)*num_images_on_page, 
      limit: page * num_images_on_page,
      text_query: text_query 
    });

    // Display files from the folder
    socket.on('emit_images_page_show_files', (data) => {
      console.log('emit_images_page_show_files', data);

      // Create a container for the images
      let container = `<div class="fixed-grid has-6-cols is-gap-0.5">
        <div class="grid" id="images_grid_container">
        </div>
      </div>`;
      // Add the container to the body (or another container)
      $('#images_preview_container').append(container);

      // Create a new div for each image
      data["files_data"].forEach(item => {
        const imageDataDiv = document.createElement('div');
        imageDataDiv.className = 'cell has-background-light p-1';

        const imageContainer = document.createElement('div');
        // Make image container always square
        imageContainer.style.aspectRatio = 1;
        // Horizontally center the image
        imageContainer.style.display = 'flex';
        imageContainer.style.justifyContent = 'center'; 
        $(imageContainer).addClass('mb-2');


        imageDataDiv.append(imageContainer);

        // <a href="https://cdn.photoswipe.com/photoswipe-demo-images/photos/2/img-2500.jpg" 
        //   data-pswp-width="1669" 
        //   data-pswp-height="2500" 
        //   target="_blank">
        //   <img src="https://cdn.photoswipe.com/photoswipe-demo-images/photos/2/img-200.jpg" alt="" />
        // </a>

        // Create an object for holding the image data
        const image = document.createElement('img');
        image.src = 'image_files/'+item.file_path;
        image.onload = function() {
          // Create link to full image
          const link = document.createElement('a');
          link.href = 'image_files/'+item.file_path;
          link.target = '_blank';
          link.setAttribute('data-pswp-width', image.width);
          link.setAttribute('data-pswp-height', image.height);

          // Reduce the size of the image to reduce the load on the page
          reducedImage = resize_image(image, 400, 400);
          reducedImage.style.maxWidth = '100%'; // Ensure the image takes the maximum amount of space
          reducedImage.style.maxHeight = '100%'; // without exceeding the square's bounds
          reducedImage.style.objectFit = 'contain'; // Maintain the aspect ratio    
          
          // Append the reduced image to the link
          link.append(reducedImage);

          // Append the image to the beginning of div
          imageContainer.prepend(link);
          image.remove();
        };

        // Add name of the file to the div
        const data = document.createElement('p');
        data.style.wordBreak = 'break-all'; // break long words
        
        data.innerHTML = '<b>Path:</b> ' + item.file_path;
        data.innerHTML += '<br><b>Hash:</b> ' + item.hash;
        data.innerHTML += '<br><b>User rating:</b> ' + item.user_rating;
        data.innerHTML += '<br><b>Model rating:</b> ' + item.model_rating;
        data.innerHTML += '<br><b>File size:</b> ' + item.file_size;
        data.innerHTML += '<br><b>Resolution:</b> ' + item.resolution;
        data.innerHTML += '<br><br>';

        // Create buttons for opening and deleting the file
        btn_open = document.createElement('button');
        btn_open.className = 'button is-pulled-left';
        btn_open.innerHTML = '<span class="icon"><i class="fas fa-folder-open"></i></span><span>Open</span>';
        btn_open.onclick = function() {
          console.log('Open file in folder: ' + item.full_path)
          socket.emit('emit_images_page_open_file_in_folder', item.full_path);
        };
        data.append(btn_open);

        // Create a button for deleting the file
        btn_delete = document.createElement('button');
        btn_delete.className = 'button is-pulled-right';
        btn_delete.innerHTML = '<span class="icon"><i class="fas fa-trash"></i></span><span>Delete</span>';
        btn_delete.onclick = function() {
          console.log('Delete file: ' + item.full_path)
          socket.emit('emit_images_page_send_file_to_trash', item.full_path);
          // refresh page
          location.reload();
        };
        data.append(btn_delete);

        //name.className = 'has-text-centered';
        imageDataDiv.append(data);

        $('#images_grid_container').append(imageDataDiv);
        window.photoGalleryLightbox.init();
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
      let folderRepresentation = create_folder_representation(data["folders"], data["folder_path"]);

      // Add the folder representation to the page
      $('#folders_menu').html(folderRepresentation);

    });

    // Display current search status
    socket.on('emit_images_page_show_search_status', (status) => {
      $('.image-search-status').html(status);
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
      url.search = params.toString();
      window.location.href = url.toString();
    });

    $('.set_search').click(function() {
      $('#search_input').val($(this).text());
      $('#seach_button').click();
    });
  })

  //// RESPONDS TO SOCKET EVENTS
  //socket.on('emit_music_page_add_radio_state', (state) => {
  //  add_radio_state(state)
  //});
})();