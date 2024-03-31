//import * from './utils.js';

// THIS METHODS SHOULD BE IMPORTED FROM utils.js
function replaceSpecialSymbols(inputText) {
  const replacedText = inputText
    .replace(/\n/g, '<br>')   // Replace newline with <br> tag
    .replace(/\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;');  // Replace tab with 4 non-breaking spaces
  return replacedText;
}

function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}

function createStarString(N) {
  if (!Number.isInteger(N)) return N;
  const blackStars = '★'.repeat(N);
  const whiteStars = '☆'.repeat(10 - N);
  return blackStars + whiteStars;
}

// Create a closed scope to avoid any variable collisions  
(function() {
  //// CONSTANTS AND VARIABLES
  let DEFAULT_COVER_IMAGE = "https://bulma.io/images/placeholders/128x128.png";

  let state_index = 0;
  let item_stack = [];
  let item_index = 0;
  
  let audioPlayer = null;
  let AIDJ_speaker = null;
  let bg_audio = null;
  let play_history = [];

  let current_audio_element = null;
  let cur_song_score = 0;
  let cur_song_hash = null;

  //// BEFORE PAGE LOADED
  function changeBackgroundVolume(startVolume, targetVolume, duration) {
    const steps = 10; // Number of steps
    const intervalTime = duration / steps;
    const volumeChange = (targetVolume - startVolume) / steps;

    let currentVolume = startVolume;
    let currentStep = 0;

    const interval = setInterval(function () {
      currentVolume += volumeChange;
      bg_audio.volume = currentVolume;

      currentStep += 1
      if (currentStep >= steps) {
        clearInterval(interval);
      }
    }, intervalTime);
  }

  let show_song_rating_timeout = null;
  function show_song_rating(score, on_hover = false){
    //console.log(score)
    if (!on_hover) cur_song_score = score;

    $('#song_rating').text('')
    let symbol = on_hover ? '•' : '◦';
    $('#song_rating').append(`<span class="is-clickable" onmouseover="show_song_rating(0, true)" onmouseup="set_song_rating(0)">${symbol}</span>`)
    for (let i = 1; i <= 10; i++) {
      let symbol = score >= i ? '★' : '☆';
      $('#song_rating').append(`<span class="is-clickable" onmouseover="show_song_rating(${i}, true)" onmouseup="set_song_rating(${i})">${symbol}</span>`)
    }

    delete(show_song_rating_timeout)
    show_song_rating_timeout = setTimeout(() => {
      show_song_rating(cur_song_score)
    }, 500);
  }
  window.show_song_rating = show_song_rating;

  function play_next_item_from_item_stack(index){
    if (index != null) item_index = index;

    if (item_index >= item_stack.length){
      console.log('Not yet generated. Waiting...')
      setTimeout(play_next_item_from_item_stack, 500);
  
      return;
    }
  
    let item = item_stack[item_index];
    console.log('play item:', item)
    $(".card").removeClass("has-background-light");
    $(`#state_${item['state_index']}`).addClass('has-background-light')
  
    $(`#state_${item['state_index']}`)[0].scrollIntoView({
      behavior: 'smooth',
      block: 'center',
      inline: 'center'
    });
  
    if (typeof item.audio_element  === 'string') {// In case it is an AI DJ audio
      console.log('AI DJ!!')
      cur_song_hash = null;
      show_song_rating(0);
  
      let file_path = item.audio_element;
      
      $("#song_label").text(`AI DJ`)
      $('#song_cover_image').attr("src", item.image);
      audioPlayer.setAttribute('src', file_path);
      audioPlayer.currentTime = 0;
      audioPlayer.play();
  
      changeBackgroundVolume(0, 0.3, 1000);
    } else { // In case it is a song
      cur_song_hash = item.audio_element['hash'];
  
      let user_rating = parseInt(item.audio_element['user_rating']) || 0;
      show_song_rating(user_rating)
  
      let file_path = item.audio_element['url_path'];
      $("#song_label").text(`${item.audio_element['artist']} - ${item.audio_element['title']} | ${item.audio_element['album']}`)
      $('#song_cover_image').attr("src", item.image); //when it will be passed to frontend as base64
      audioPlayer.setAttribute('src', file_path);
      audioPlayer.currentTime = 0;
      audioPlayer.play();
  
      changeBackgroundVolume(0.3, 0, 1000);
    }
    audio_play_song()
    item_index += 1
  }

  function add_radio_state(state){
    console.log('New state:', state)
    
    let image_template = '';
    if ('image' in state) {
      const image_url = state['image'] || DEFAULT_COVER_IMAGE;
      image_template = `<figure class="media-left mb-0 ml-0 mr-5">
                          <p class="image is-96x96">
                            <img src="${image_url}">
                          </p>
                        </figure>`
    }
  
    $("#tab_content_radio").append(
      `<div id="state_${state_index}" class="card is-clickable" ${state['hidden'] ? 'style="opacity: 30%; {{"" if cfg.music_page.show_hidden_messages else "display:none" }}"': ''}>
        <div class="card-content media">
          ${image_template}
          <div class="media-content">
            <div class="content">
              <p>
                <strong>${state['head']}</strong>
                <br>
                ${replaceSpecialSymbols(state['body'])}
              </p>
            </div>
          </div>
        </div>
      </div>`)
  
    if ('audio_element' in state) {
      let item_index = item_stack.length;
      item_stack.push({
        "state_index": state_index,
        "audio_element": state['audio_element'],
        "image": state['image']
      })
  
      $(`#state_${state_index}`).click(()=>{
        play_next_item_from_item_stack(item_index)
      })

      let cur_song_play_time = localStorage.getItem("music_page_song_play_time");
      let cur_song_hash = localStorage.getItem("music_page_song_hash");
      //let cur_item_index = localStorage.getItem("music_page_song_index");

      if (cur_song_hash == state['audio_element']['hash']) {
        $("#block_start_radio_session").hide();
        play_next_item_from_item_stack(item_index);
        audioPlayer.currentTime = cur_song_play_time;      
      } 
      
      if (!cur_song_hash && item_stack.length == 1) {
        play_next_item_from_item_stack()
      }
    }
  
    state_index += 1
  }

  function set_song_rating(score){
    console.log('set_song_rating', score)
    cur_song_score = score;
    show_song_rating(score)
    $(`#play_table_user_rating_${cur_song_hash}`).text(createStarString(score));
  
    socket.emit('emit_music_page_set_song_rating', {"hash": cur_song_hash, "score": score});
  }
  window.set_song_rating = set_song_rating;
  
  // AUDIO CONTROL METHODS
  
  function play_next_track(){
    if (cur_song_hash != null) socket.emit('emit_music_page_set_song_play_rate', [cur_song_hash, -1]);
  
    audioPlayer.pause();
    bg_audio.pause();
    play_next_item_from_item_stack();
  }
  
  function play_previous_track(){
    //if (play_history.length>1)
    //  play_history.pop()
  
    //if (play_history.length>0)
    //  play_song(...play_history.pop())
  }
  
  function audio_play_song(){
    console.log('audioPlayer', audioPlayer.getAttribute("src"))
    if (audioPlayer.getAttribute("src") == null) {
      socket.emit('emit_music_page_get_next_song', [])
      audioPlayer.pause();
      //bg_audio.currentTime = 0;
      //bg_audio.play()
    } else {
      socket.emit('emit_music_page_song_start_playing', cur_song_hash);
      //bg_audio.pause()
      bg_audio.play();
      audioPlayer.play();
  
      $("#play_btn").find('i').removeClass('fa-play');
      $("#play_btn").find('i').addClass('fa-pause');
    }
  }
  
  function audio_pause_song(){
    bg_audio.pause();
    audioPlayer.pause()
  
    $("#play_btn").find('i').removeClass('fa-pause');
    $("#play_btn").find('i').addClass('fa-play');
  }
  
  function audio_stop_song(){
    bg_audio.stop();
    audioPlayer.stop()
  }

  //// AFTER PAGE LOADED
  $(document).ready(function() {
    // Set up audio elements
    audioPlayer = $('#my_audio')[0];
    AIDJ_speaker = $('#AIDJ_speaker')[0];
    bg_audio = $('#bg_audio')[0];

    bg_audio.setAttribute('src', `static/background-music/angelic-pad-loopwav-14643.mp3?=${getRandomInt(999999)}`);
    bg_audio.loop=true;
    bg_audio.volume = 0.3;
    bg_audio.pause()

    // Request next song if the previous has ended
    audioPlayer.onended = function() {
      if (cur_song_hash != null) socket.emit('emit_music_page_set_song_play_rate', [cur_song_hash, +1]);
      audioPlayer.pause();
      play_next_item_from_item_stack();
    };

    // Update progress bar to see how much of the current song has been played
    audioPlayer.addEventListener("timeupdate", function() {
      var currentTime = audioPlayer.currentTime;
      var duration = audioPlayer.duration;
      //$('.hp_range').stop(true,true).animate({'width':(currentTime +.25)/duration*100+'%'},250,'linear');

      if (duration>0)
        $("#song_progress").val((currentTime +.25)/duration*100)
      else
        $("#song_progress").val(0)

      localStorage.setItem("music_page_song_play_time", currentTime);
      localStorage.setItem("music_page_song_hash", cur_song_hash);
      //localStorage.setItem("music_page_song_index", item_index);
    });

    // Add a click event listener to the progress bar
    const progressBar = $("#song_progress");
    progressBar.on("click", function (event) {
      // Calculate the click position as a percentage of the total width
      const clickX = event.clientX - progressBar.get(0).getBoundingClientRect().left;
      const progressBarWidth = progressBar.get(0).clientWidth;
      const clickPercentage = (clickX / progressBarWidth) * 100;

      // Update the progress bar value
      progressBar.val(clickPercentage);

      // Update the audio player's playback position
      const audioDuration = audioPlayer.duration;
      const newPosition = (clickPercentage / 100) * audioDuration;
      audioPlayer.currentTime = newPosition;
    });

    $("#prev_btn").click(()=>{ 
      play_previous_track();
    });
    $("#play_btn").click(()=>{
      console.log('play_btn click')
      if (audioPlayer.paused)
        audio_play_song()
      else
        audio_pause_song()
    });
    $("#next_btn").click(()=>{
      play_next_track();
    });

    $(`#song_edit_btn`).click(()=>{ 
      let item = item_stack[item_index-1];
      if (typeof item.audio_element === 'string') return;

      $(`#song_edit_modal`).addClass('is-active')
      $('#song_edit_content').html('')

      //metadata['duration'] = tag.duration #(seconds)
      //metadata['bitrate'] = tag.bitrate #(kbps)
      function generate_field_template(field_name, field_key, type="text"){
        return `<div class="field has-addons">
                  <p class="control mb-0">
                    <a class="button is-static">
                      ${field_name}:
                    </a>
                  </p>
                  <p class="control is-expanded">
                    <input id="song_edit_${field_key}" class="input" type="${type}" value="${item.audio_element[field_key]}">
                  </p>
                </div>`
      }

      let template = `<div class="card-content">
        <p>File path: ${item.audio_element['file_path']} </p>
        <p>Url path: ${item.audio_element['url_path']} </p>
        <p>Audio-based-Hash: ${item.audio_element['hash']} </p>
        ${generate_field_template('Artist', 'artist')}
        ${generate_field_template('Title', 'title')}
        ${generate_field_template('Album', 'album')}
        ${generate_field_template('Year', 'date', 'number')}
        ${generate_field_template('Track number', 'track_num', 'number')}
        ${generate_field_template('Ganre', 'genre')}
        <p>Lyrics:</p>
        <textarea id="song_edit_lyrics" class="textarea has-fixed-size" rows="15">${item.audio_element['lyrics']}</textarea>
      </div>

      <footer class="card-footer">
        <a class="card-footer-item is-info" id="song_edit_modal_btn_update">Update song info</a>
        <a class="card-footer-item" id="song_edit_modal_btn_close">Close</a>
      </footer>`

      $('#song_edit_content').append(template);

      $('#song_edit_modal_btn_close').click((event)=>{ 
        $(`#song_edit_modal`).removeClass('is-active');
      });

      $('#song_edit_modal_btn_update').click((event)=>{ 
        item.audio_element['artist']    = $('#song_edit_artist').val(); 
        item.audio_element['title']     = $('#song_edit_title').val(); 
        item.audio_element['date']      = $('#song_edit_date').val(); 
        item.audio_element['track_num'] = $('#song_edit_track_num').val(); 
        item.audio_element['genre']     = $('#song_edit_genre').val(); 
        item.audio_element['lyrics']    = $('#song_edit_lyrics').val(); 

        let data = {
          'hash':      item.audio_element['hash'],
          'file_path': item.audio_element['file_path'],
          'artist':    $('#song_edit_artist').val(),
          'title':     $('#song_edit_title').val(),
          'album':     $('#song_edit_album').val(),
          'date':      $('#song_edit_date').val(),
          'track_num': $('#song_edit_track_num').val(),
          'genre':     $('#song_edit_genre').val(),
          'lyrics':    $('#song_edit_lyrics').val(),
        };

        socket.emit('emit_music_page_update_song_info', data);
        $(`#song_edit_modal`).removeClass('is-active');
      });
    });

    $( "#song_rating" ).mouseleave(function() {
      show_song_rating(cur_song_score)
    });

    if ('mediaSession' in navigator) {
    //setting the metadata
      navigator.mediaSession.metadata = new MediaMetadata({
        title: 'Unforgettable',
        artist: 'Nat King Cole',
        album: 'The Ultimate Collection (Remastered)',
        artwork: [
          { src: 'https://dummyimage.com/96x96',   sizes: '96x96',   type: 'image/png' },
          { src: 'https://dummyimage.com/128x128', sizes: '128x128', type: 'image/png' },
          { src: 'https://dummyimage.com/192x192', sizes: '192x192', type: 'image/png' },
          { src: 'https://dummyimage.com/256x256', sizes: '256x256', type: 'image/png' },
          { src: 'https://dummyimage.com/384x384', sizes: '384x384', type: 'image/png' },
          { src: 'https://dummyimage.com/512x512', sizes: '512x512', type: 'image/png' },
        ]
      });
      
      navigator.mediaSession.setActionHandler('play', function() { 
        audio_play_song()
      });
      navigator.mediaSession.setActionHandler('pause', function() { 
        audio_pause_song()
      });
      navigator.mediaSession.setActionHandler('stop', function() { 
        audio_stop_song()
      });
      navigator.mediaSession.setActionHandler('seekbackward', function() {  });
      navigator.mediaSession.setActionHandler('seekforward', function() {  });
      navigator.mediaSession.setActionHandler('seekto', function() {  });
      navigator.mediaSession.setActionHandler('previoustrack', function() { 
        play_previous_track();
      });
      navigator.mediaSession.setActionHandler('nexttrack', function() { 
        play_next_track();
      });
      //navigator.mediaSession.setActionHandler('skipad', function() {  });
    }

    //// ADD RADIO REGULATORS
    $("#tab_content_radio").append(
      `<div class="block" id="block_start_radio_session">
        <div class="field">
          <label class="label">What music would you like to listen today?</label>
          <div class="control">
            <input class="input" id="radio_session_prompt" type="text" placeholder="Enter music genre, artist, song or anything else you have in mind">
          </div>
        </div>

        <div class="field">
          <input id="radio_session_use_AIDJ" type="checkbox" name="switchRoundedDefault" class="switch is-rounded" checked="checked">
          <label for="radio_session_use_AIDJ">Activate AI DJ</label>
        </div>

        <div class="field">
          <div class="control">
            <button class="button is-primary is-fullwidth" id="radio_session_start">Start Radio Session</button>
          </div>
        </div>
      </div>`)

    $("#radio_session_start").click(()=>{
      localStorage.removeItem("music_page_song_play_time");
      localStorage.removeItem("music_page_song_hash");

      socket.emit('emit_music_page_radio_session_start', {
        "prompt": $("#radio_session_prompt").val(),
        "use_AIDJ": $("#radio_session_use_AIDJ").is(":checked")
      });
      $("#block_start_radio_session").hide();
    });

    // Request the current state of radio history
    socket.emit('emit_music_page_get_radio_history')
  })

  //// RESPONDS TO SOCKET EVENTS
  socket.on('emit_music_page_show_radio_history', (radio_states) => {
    radio_states.forEach((state, ind) => {
      add_radio_state(state)
    })
  })

  socket.on('emit_music_page_add_radio_state', (state) => {
    add_radio_state(state)
  });

  // Read next song data and play it
  socket.on('emit_music_page_send_next_song', (audio_element) => {
    let file_path = audio_element['url_path']
    let hash = audio_element['hash'];

    /*$("table tr.is-selected").removeClass("is-selected has-background-link-dark");
    $(`#play_${hash}`).addClass("is-selected has-background-link-dark")
    console.log(file_path)
    //audioPlayer.pause();

    $("#song_label").text(`${audio_element['artist']} - ${audio_element['title']} | ${audio_element['album']}`)

    show_song_rating(4)
    
    audioPlayer.setAttribute('src', file_path);
    audioPlayer.currentTime = 0;
    audioPlayer.play();*/
    AIDJ_speaker.setAttribute('src', `static/tmp/AIDJ.wav?=${getRandomInt(999999)}`);
    AIDJ_speaker.currentTime = 0;
    AIDJ_speaker.play();
    AIDJ_speaker.onended = function() {
      play_song(file_path, hash, audio_element)
    }
  })

  // Add a button to request more songs if the session has ended
  socket.on('emit_music_page_radio_session_end', (state) => {
    //// ADD RADIO REGULATORS
    $("#tab_content_radio").append(
      `<div class="block" id="block_continue_radio_session">
        <div class="field">
          <div class="control">
            <button class="button is-primary is-fullwidth" id="radio_session_continue">Continue Radio Session</button>
          </div>
        </div>
      </div>`)

    $("#radio_session_continue").click(()=>{
      socket.emit('emit_music_page_radio_session_start', {
        "prompt": $("#radio_session_prompt").val(),
        "use_AIDJ": $("#radio_session_use_AIDJ").is(":checked")
      });
      $("#block_continue_radio_session").remove();
    });
  });
})();