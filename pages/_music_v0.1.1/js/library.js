// THIS METHODS SHOULD BE IMPORTED FROM utils.js
//Random string generator
function makeid(length) {
  var text = "";
  var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

  for (var i = 0; i < length; i++)
    text += possible.charAt(Math.floor(Math.random() * possible.length));

  return text;
}

// Create a closed scope to avoid any variable collisions  
(function() {
  //// BEFORE PAGE LOADED

  //// AFTER PAGE LOADED
  $(document).ready(function() {
    // Request current music library folder path
    socket.emit('emit_music_page_get_path_to_music_folder');

    // Request files from the main media folder
    socket.emit('emit_music_page_get_files', '');
    
    $('#library_files_breadcrumb').empty();
    $('#library_files_breadcrumb').append(
      `<li class="is-active"><a href="#" aria-current="page">media</a></li>`
    );
    
    // Display files from the folder
    socket.on('emit_music_page_show_files', (data) => {
      console.log(data)

      // Update breadcrumb
      $('#library_files_breadcrumb').empty();

      let indx = makeid(16);
      $('#library_files_breadcrumb').append(
        `<li><a id="${indx}">media</a></li>`
      );
      $(`#${indx}`).click(()=>{ 
        socket.emit('emit_music_page_get_files', '');
      });

      data["folder_path"].split('/').forEach((element, ind, array) => {
        let indx = makeid(16);
        let path = array.slice( 0, ind + 1 ).join( '/' );
        $('#library_files_breadcrumb').append(
          `<li><a id="${indx}">${element}</a></li>`
        );

        $(`#${indx}`).click(()=>{ 
          socket.emit('emit_music_page_get_files', path);
        });
      })

      // Show current files 
      $('#library_files_container').empty();

      data["files_data"].forEach((element, ind) => {
        let indx = makeid(16);
        $('#library_files_container').append(
          `<div id="${indx}" class="column is-2 is-clickable is-centered">
            <div class="box">
              <i class="fa-solid fa-${element["type"]} fa-8x"></i>
              <br><br>
              ${element["base_name"]}
            </div>
          </div>`
        );

        $(`#${indx}`).click(()=>{ 
          socket.emit('emit_music_page_get_files', element["file_path"]);
        });
      });
    })

    // Show current music library folder path
    socket.on('emit_music_page_show_path_to_music_folder', (current_path) => {
      $('#path_to_music_folder').val(current_path);
    });

    // Start update of the music library
    $(`#update_music_library`).click(()=>{ 
      socket.emit('emit_music_page_update_music_library');
      $('#music_library_update_status').removeClass('fa-circle-check').addClass('fa-spinner fa-pulse');
    });

    // Update path to the music library folder
    $(`#update_path_to_music_folder`).click(()=>{ 
      socket.emit('emit_music_page_update_path_to_music_folder', $('#path_to_music_folder').val());
    });
  })

  // Update the progress bar to show the progress of the music library update
  socket.on('emit_music_page_update_music_library_progress', (percent) => {
    $("#music_library_update_progress").val(percent);
    if (percent == 100) {
      $('#music_library_update_status').removeClass('fa-spinner fa-pulse').addClass('fa-circle-check');
    } else {
      $('#music_library_update_status').removeClass('fa-circle-check').addClass('fa-spinner fa-pulse');
    }
  });
})();

