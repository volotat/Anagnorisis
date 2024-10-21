// THIS METHODS SHOULD BE IMPORTED FROM utils.js

class StarRatingComponent {
  constructor({ initialRating = Name, callback = ()=>{} }) {
    this.maxRating = 10;
    this.rating = initialRating;
    this.containers = [];
    this.callback = callback;
  }

  issueNewHtmlComponent(params) {
    const starRatingHTMLContainer = new StarRatingHTMLContainer(this, params);
    this.containers.push(starRatingHTMLContainer);

    return starRatingHTMLContainer.container;
  }

  updateAllContainers() {
    this.containers.forEach(container => container.updateDisplay());
  }
}

class StarRatingHTMLContainer {
  constructor(starRatingObject, {containerType = 'div', size = 3, isActive = false, showPassiveAsNumber = true}) {
    this.starRatingObject = starRatingObject;
    this.isActive = isActive;
    this.showPassiveAsNumber = showPassiveAsNumber;
    this.symbolsList = [];
    this.container = document.createElement(containerType);
    
    this.container.classList.add(`is-size-${size.toString()}`);
    this.container.classList.add('is-gapless');
    this.container.classList.add('has-text-centered');
    this.container.classList.add('is-unselectable');
  
    this.updateDisplay();
  }

  generateStarDisplay() {
    const starRatingObject = this.starRatingObject;

    // Add the initial symbol based on the rating
    const initialSymbol = document.createElement('span');
    initialSymbol.textContent = starRatingObject.rating === null ? '◦' : '•';
    this.container.appendChild(initialSymbol);
    this.symbolsList.push(initialSymbol);
  
    // Create each star element
    for (let i = 1; i <= starRatingObject.maxRating; i++) {
      const star = document.createElement('span');
      star.textContent = i <= starRatingObject.rating ? '★' : '☆';
      star.classList.add('star');
  
      this.container.appendChild(star);
      this.symbolsList.push(star);
    }

    if (this.isActive) {
      for (let i = 0; i < this.symbolsList.length; i++) {
        this.symbolsList[i].classList.add('is-clickable');

        this.symbolsList[i].addEventListener('mouseover', () => {
          this.updateDisplay(i); 
        });

        this.symbolsList[i].addEventListener('mouseout', () => {
          this.updateDisplay(); 
        });

        this.symbolsList[i].addEventListener('click', () => {
          starRatingObject.rating = i;
          starRatingObject.callback(i);
          starRatingObject.updateAllContainers();
        });
      }
    }
  }

  updateDisplay(tmpRating = null) {
    let rating = this.starRatingObject.rating;
    const maxRating = this.starRatingObject.maxRating;

    if (tmpRating != null) rating = tmpRating;

    if (!this.isActive && this.showPassiveAsNumber) {
      if (rating == null)
        this.container.innerHTML = 'Not rated yet';
      else
        this.container.innerHTML = rating.toString() + '/' + maxRating.toString();

      // Clear the symbols list in case it was previously active for some reason
      this.symbolsList = [];
    } else {
      if (this.symbolsList.length == 0) {
        this.generateStarDisplay();
      }

      this.symbolsList[0].textContent = rating === null ? '◦' : '•';
      for (let j = 1; j <= maxRating; j++) {
        this.symbolsList[j].textContent = j <= rating ? '★' : '☆';
      }
    }
  }
}




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
      console.log('active_path', active_path);
      console.log('isSubfolder', isSubfolder(active_path, current_path_));
      const isActive = active_path === current_path_ ? 'is-active' : '';
      let encoded_link = `path=${encodeURIComponent(current_path_)}`;
      let color = isSubfolder(active_path, current_path_) ? 'has-text-black' : 'has-text-grey-light';
      if (active_path === current_path_) color = '';
      folderRepresentation += `<li><a class="${isActive} ${color}" href="?${encoded_link}">${folder}</a>`;
      
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
    // Request current media folder path
    socket.emit('emit_images_page_get_path_to_media_folder');

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
        imageContainer.classList.add('pswp-gallery__item')
        // Make image container always square
        imageContainer.style.aspectRatio = 1;
        // Horizontally center the image
        imageContainer.style.display = 'flex';
        imageContainer.style.justifyContent = 'center'; 
        $(imageContainer).addClass('mb-2');


        imageDataDiv.append(imageContainer);

        // Create a new star rating component
        const callback = (rating) => {
          console.log('New rating:', rating);
          socket.emit('emit_images_page_set_image_rating', {
            hash: item.hash,
            file_path: item.file_path,
            rating: rating,
          });
        }
        const starRating = new StarRatingComponent({
          callback: callback,
          initialRating: item.user_rating, //parseInt(Math.random() * 11),
        });

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

          
          // Create a new div element
          const hiddenCaptionDiv = document.createElement('div');
          hiddenCaptionDiv.className = 'hidden-caption-content';

          // Set the innerHTML of the div to include the file path or any other HTML content
          let starRatingDiv = starRating.issueNewHtmlComponent({
            containerType: 'div',
            size:3, 
            isActive: true
          });
          hiddenCaptionDiv.append(starRatingDiv);
          // Append the div to the imageContainer
          imageContainer.append(hiddenCaptionDiv);
          

          //$(imageContainer).append('<div class="hidden-caption-content">' + item.file_path + '</div>');
          image.remove();
        };

        // Add name of the file to the div
        const data = document.createElement('p');
        data.style.wordBreak = 'break-all'; // break long words

        $(data).append('<b>Path:</b>&nbsp;' + item.file_path + '<br>');
        $(data).append('<b>Hash:</b>&nbsp;' + item.hash + '<br>');
        

        const StarRatingComponentObject = starRating.issueNewHtmlComponent({
          containerType: 'span',
          size:6, 
          isActive: false
        })
        $(data).append('<br><b>User rating:</b>&nbsp;&nbsp;');
        $(data).append(StarRatingComponentObject)
        $(data).append('<br>');

        if (item.model_rating != null){
          $(data).append('<b>Model rating:</b>&nbsp;' + item.model_rating.toFixed(2) + '/10<br>');
        } else {
          $(data).append('<b>Model rating:</b>&nbsp;N/A<br>');
        }
        
        $(data).append('<b>File size:</b>&nbsp;' + item.file_size + '<br>');
        $(data).append('<b>Resolution:</b>&nbsp;' + item.resolution + '<br><br>');

        
        /*data.innerHTML = '<b>Path:</b> ' + item.file_path;
        data.innerHTML += '<br><b>Hash:</b> ' + item.hash;
        data.innerHTML += '<br><b>User rating:</b> ' + starRating.issueNewHtmlComponent({
          size:1, 
          isActive: false
        }).innerHTML; //+ item.user_rating;
        data.innerHTML += '<br><b>Model rating:</b> ' + item.model_rating;
        data.innerHTML += '<br><b>File size:</b> ' + item.file_size;
        data.innerHTML += '<br><b>Resolution:</b> ' + item.resolution;
        data.innerHTML += '<br><br>';*/

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
        //window.photoGalleryLightbox.init();
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
    socket.on('emit_images_page_show_search_status', (status) => {
      $('.image-search-status').html(status);
    });

    // Show current media folder path
    socket.on('emit_images_page_show_path_to_media_folder', (current_path) => {
      $('#path_to_media_folder').val(current_path);
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
  })

  //// RESPONDS TO SOCKET EVENTS
  //socket.on('emit_music_page_add_radio_state', (state) => {
  //  add_radio_state(state)
  //});
})();