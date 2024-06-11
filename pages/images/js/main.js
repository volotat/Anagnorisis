// Create a closed scope to avoid any variable collisions  
(function() {
  //// CONSTANTS AND VARIABLES
  let num_images_on_page = 50;

  //// BEFORE PAGE LOADED
  // read page number from the URL ?page=1
  const urlParams = new URLSearchParams(window.location.search);
  let pageParam = parseInt(urlParams.get('page'));
  let page = (!pageParam || pageParam < 1) ? 1 : pageParam;
  let text_query = urlParams.get('text_query') || '';
  text_query = decodeURIComponent(text_query);

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

  //// AFTER PAGE LOADED
  $(document).ready(function() {
    // Request files from the main media folder
    socket.emit('emit_images_page_get_files', {
      path: '', 
      pagination: (page-1)*num_images_on_page, 
      limit: page * num_images_on_page,
      text_query: text_query 
    });

    // Display files from the folder
    socket.on('emit_images_page_show_files', (data) => {
      // Create a container for the images
      let container = `<div class="fixed-grid has-5-cols is-gap-0.5">
        <div class="grid" id="images_grid_container">
        </div>
      </div>`;
      // Add the container to the body (or another container)
      $('#images_preview_container').append(container);

      // Create a new div for each image
      data["files_data"].forEach(item => {
        const imageDiv = document.createElement('div');
        imageDiv.className = 'cell has-background-light p-1';

        const image = document.createElement('img');
        image.src = 'images_files/'+item.file_path;
        image.onload = function() {
          reducedImage = resize_image(image, 400, 400);
          imageDiv.appendChild(reducedImage);
          image.remove();
        };

        $('#images_grid_container').append(imageDiv);
      });

      // update the pagination
      $("#pagination_list").empty();
      let total_pages = Math.ceil(data["total_files"] / num_images_on_page);

      for (let i = 1; i <= total_pages; i++) {
        // only include the first, one before, one after, and last pages
        if (i == 1 || i == page - 1 || i == page || i == page + 1 || i == total_pages) {
          let link = `page=${i}`;
          
          if (text_query){
            let encoded_text_query = encodeURIComponent(text_query);
            link = link + `&text_query=${encoded_text_query}`
          }

          let template = `<li>
            <a href="?${link}" class="pagination-link ${i == page?'is-current':''}" aria-label="Goto page ${i}">${i}</a>
          </li>`
          $("#pagination_list").append(template);
        }
        // add ellipsis when there are skipped pages
        else if (i == 2 && page > 3 || i == total_pages - 1 && page < total_pages - 2) {
          let template = `<li>
            <span class="pagination-ellipsis">&hellip;</span>
          </li>`
          $("#pagination_list").append(template);
        }
      }
    })

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
  })

  //// RESPONDS TO SOCKET EVENTS
  //socket.on('emit_music_page_add_radio_state', (state) => {
  //  add_radio_state(state)
  //});
})();