class PlaylistManager {
    constructor(audioPlayer, playlistElement, song_cover_image_element, song_label_element, socket) {
        this.audioPlayer = audioPlayer;
        this.playlistElement = playlistElement;
        this.song_cover_image_element = song_cover_image_element;
        this.song_label_element = song_label_element;
        this.socket = socket;
        this.playlist = [];
        this.currentSongIndex = 0;
        this.DEFAULT_COVER_IMAGE = "static/images/128x128.png";
        this.isPlaying = false;
        this.MAX_SONGS_SHOWN = 30;
    }

    updatePlaylistPanel() {
        // Use a starting index (for display) and slice your playlist
        let startIndex = Math.max(this.currentSongIndex - parseInt(this.MAX_SONGS_SHOWN / 2), 0);
        let playlistShown = this.playlist.slice(startIndex, startIndex + this.MAX_SONGS_SHOWN);

        let playlistHtml = '';
        playlistShown.forEach((item, index) => {
          // Get the song's real position in the full playlist
          let globalIndex = startIndex + index;
      
          // Highlight if it's the currently playing song
          let isActiveClass = (globalIndex === this.currentSongIndex) ? 'is-active' : '';
      
          playlistHtml += `
            <a class="panel-block ${isActiveClass}" data-path="${item.file_path}" data-index="${globalIndex}">
              <span class="panel-icon is-size-7">
                <i class="fas fa-music" aria-hidden="true"></i>
              </span>
              ${item.base_name}
            </a>
          `;
        });
      
        // Show how many songs are left if truncated
        if (this.playlist.length - startIndex - this.MAX_SONGS_SHOWN > 0) {
          playlistHtml += `
            <a class="panel-block">
              ${this.playlist.length - startIndex - this.MAX_SONGS_SHOWN} more songs...
            </a>
          `;
        }
      
        // Insert the HTML into the panel
        this.playlistElement.html(playlistHtml);
  
        // Add click event for each panel-block
        this.playlistElement.find('.panel-block').on('click', (event) => {
          console.log('clicked');
          // Parse the global index of the clicked song
          if (!$(event.currentTarget).data('index')) return;

          let clickedIndex = parseInt($(event.currentTarget).data('index'));
          this.playSongAtIndex(clickedIndex);
        });
    }

    setPlaylist(all_files_paths){
        this.playlist = all_files_paths;
        this.playlist = this.playlist.map((item) => {
            return {
              file_path: item,
              base_name: item.split('/').pop()
            };
        });
        
        this.currentSongIndex = 0;
        this.updatePlaylistPanel();
    }
    
    playSongAtIndex(index = 0) {
        this.currentSongIndex = index;
        const currentSong = this.playlist[this.currentSongIndex];
        this.audioPlayer.src = '/music_files/' + currentSong.file_path;
        this.audioPlayer.play();
        this.isPlaying = true;
        this.updateSongInfo(currentSong);
        this.updatePlaylistPanel();
    }
    
    nextSong(){
        if(this.playlist.length == 0) return;
        this.playSongAtIndex((this.currentSongIndex + 1) % this.playlist.length);
    }

    previousSong(){
        if(this.playlist.length == 0) return;
        this.playSongAtIndex((this.currentSongIndex - 1 + this.playlist.length) % this.playlist.length);
    }

    pauseSong(){
        this.audioPlayer.pause();
        this.isPlaying = false;
    }

    playSong(){
        if (this.audioPlayer.getAttribute("src") == null) {
           this.playSongAtIndex(0);
           return;
        }

        this.audioPlayer.play();
        this.isPlaying = true;
    }

    togglePlay(){
        if(this.isPlaying) {
            this.pauseSong();
        }
        else {
            this.audioPlayer.play();
            this.isPlaying = true;
        }
    }

    updateSongInfo(song){
        console.log(song);
        // this.song_label_element.text(`${song.audiofile_data['artist']} - ${song.audiofile_data['title']} | ${song.audiofile_data['album']}`);
        this.song_label_element.text(`${song.base_name}`);
        //if (song.audiofile_data['image'] == null)
        //    this.song_cover_image_element.attr("src", this.DEFAULT_COVER_IMAGE);
        //else
        //    this.song_cover_image_element.attr("src", song.audiofile_data['image']);
    }
}

export default PlaylistManager;