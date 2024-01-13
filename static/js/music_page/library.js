// Create a closed scope to avoid any variable collisions  
(function() {
  //// BEFORE PAGE LOADED

  //Random string generator
  function makeid(length) {
    var text = "";
    var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for (var i = 0; i < length; i++)
      text += possible.charAt(Math.floor(Math.random() * possible.length));

    return text;
  }

  //// AFTER PAGE LOADED
  $(document).ready(function() {
    // Request files from the main media folder
    socket.emit('emit_music_page_get_files', '');

    // Display files from the folder
    socket.on('emit_music_page_show_files', (files_data) => {
      $('#library_files_container').empty();

      console.log(files_data)
      files_data.forEach((element, ind) => {
        let indx = makeid(8);
        $('#library_files_container').append(
          `<div id="file_${indx}" class="column is-2 is-clickable is-centered">
            <div class="box">
              <i class="fa-solid fa-${element["type"]} fa-8x"></i>
              <br><br>
              ${element["base_name"]}
            </div>
          </div>`
        );

        $(`#file_${indx}`).click(()=>{ 
          socket.emit('emit_music_page_get_files', element["file_path"]);
        });
      });
    })
  })
})();

