import StarRatingComponent from '/pages/StarRating.js';

class SongControlPanel {
    constructor(audioPlayer, songCoverElement, songLabelElement, songRatingElement, songProgressElement, socket, playlistManager) {
        this.audioPlayer = audioPlayer;
        this.songCoverElement = songCoverElement;
        this.songLabelElement = songLabelElement;
        this.songRatingElement = songRatingElement;
        this.songProgressElement = songProgressElement;
        this.socket = socket;
        this.playlistManager = playlistManager;
        this.DEFAULT_COVER_IMAGE = "static/images/128x128.png";
        this.currentSongHash = null;
        this.currentSongScore = null;
        this.showSongRatingTimeout = null;
        this.starRatingComponent = null;

        this.setupEventListeners();
        this.setupRatingComponent();

        $("#prev_btn").click(()=>{ 
            this.previousSong();
        });
        $("#play_btn").click(()=>{
            this.togglePlay();
        });
        $("#next_btn").click(()=>{
            this.nextSong();
        });
    }

    setupEventListeners() {
        this.audioPlayer.addEventListener("timeupdate", () => this.updateProgressBar());
        this.audioPlayer.addEventListener("ended", () => { this.nextSong(true) });
        this.songProgressElement.on("click", (event) => this.handleProgressBarClick(event));

        if ('mediaSession' in navigator) {
            const songControlPanel = this;

            //setting the metadata
            navigator.mediaSession.metadata = new MediaMetadata({
                title: 'Unforgettable',
                artist: 'Artist',
                album: 'Album',
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
                songControlPanel.playSong()
            });
            navigator.mediaSession.setActionHandler('pause', function() { 
                songControlPanel.pauseSong()
            });
            navigator.mediaSession.setActionHandler('stop', function() { 
                songControlPanel.pauseSong()
            });
            navigator.mediaSession.setActionHandler('seekbackward', function() {  });
            navigator.mediaSession.setActionHandler('seekforward', function() {  });
            navigator.mediaSession.setActionHandler('seekto', function() {  });
            navigator.mediaSession.setActionHandler('previoustrack', function() { 
                songControlPanel.previousSong();
            });
            navigator.mediaSession.setActionHandler('nexttrack', function() { 
                songControlPanel.nextSong();
            });
            //navigator.mediaSession.setActionHandler('skipad', function() {  });
        }
    }

    setupRatingComponent() {
      const callback = (rating) => {
        this.setSongRating(rating);
      };
      this.starRatingComponent = new StarRatingComponent({callback: callback});
      let element = this.starRatingComponent.issueNewHtmlComponent({
        containerType: 'span',
        //size:3, 
        isActive: true,
      });
      this.songRatingElement.empty();
      this.songRatingElement.append(element);
    }

    updateProgressBar() {
        const currentTime = this.audioPlayer.currentTime;
        const duration = this.audioPlayer.duration;
        this.songProgressElement.val(duration > 0 ? ((currentTime + 0.25) / duration * 100) : 0);

        // Save current song hash and play time to localStorage
        //localStorage.setItem("music_page_song_hash", this.currentSongHash);
        localStorage.setItem("music_page_song_play_time", currentTime);
    }

    handleProgressBarClick(event) {
        const clickX = event.clientX - this.songProgressElement.get(0).getBoundingClientRect().left;
        const progressBarWidth = this.songProgressElement.get(0).clientWidth;
        const clickPercentage = (clickX / progressBarWidth) * 100;
        this.songProgressElement.val(clickPercentage);
        const audioDuration = this.audioPlayer.duration;
        this.audioPlayer.currentTime = (clickPercentage / 100) * audioDuration;
    }

    setSongRating(score, updateServer = true) {
        this.currentSongScore = score;
        //this.showSongRating(score);

        this.starRatingComponent.rating = score;
        this.starRatingComponent.updateAllContainers();

        if (updateServer && this.currentSongHash) {
            console.log(`Calling server to set '${this.currentSongHash}' song rating: ${score}`);
            this.socket.emit('emit_music_page_set_song_rating', { hash: this.currentSongHash, score: score });
        }
    }

    // `${item.audio_element['artist']} - ${item.audio_element['title']} | ${item.audio_element['album']}`
    // this.song_label_element.text(`${song.base_name}`);

    updateSongInfo(song) {
        // Initially, song contains only file_path info.
        this.currentSongHash = null;
        // Optionally, show a loading state here.
        this.songLabelElement.text(`Loading song details...`);

        // Now request additional info from the server.
        this.fetchSongDetails(song.file_path)
          .then((details) => {
            console.log("Song details:", details);

            // Merge detailed info into the song object.
            song.artist = details.artist;
            song.title = details.title;
            song.album = details.album;
            song.image = details.image;
            song.user_rating = details.user_rating;
            song.model_rating = details.model_rating;
            this.currentSongHash = details.hash;

            // Update UI components.
            //this.showSongRating(parseInt(song.user_rating) || parseInt(song.model_rating) || null, false, song.user_rating != null);
            this.starRatingComponent.isUserRated = song.user_rating != null;
            this.setSongRating(parseFloat(song.user_rating) || parseFloat(song.model_rating) || null, false);
            this.songLabelElement.text(`${song.artist} - ${song.title} | ${song.album}`);
            this.songCoverElement.attr("src", song.image || this.DEFAULT_COVER_IMAGE);

            // Report the server that song is playing to update the last play time.
            this.socket.emit('emit_music_page_song_start_playing', this.currentSongHash);
          })
          .catch((error) => {
            console.error("Failed to fetch song details:", error);
            this.songLabelElement.text("Error loading song details. Skipping to next song...");

            // Skip to the next song without penalizing the score
            setTimeout(() => {
                this.nextSong(false, false);
            }, 1000);
          });
    }

    fetchSongDetails(filePath) {
        return new Promise((resolve, reject) => {
            // Emit a socket event asking for song details.
            this.socket.emit('emit_music_page_get_song_details', { file_path: filePath });

            // Listen once for the response, e.g., 'emit_song_details'
            this.socket.once('emit_music_page_show_song_details', (data) => {
                if (data && data.file_path === filePath) {
                    resolve(data);
                } else {
                    reject("Song details mismatch.");
                }
            });
            // Add a timeout to reject the promise.
            setTimeout(() => reject("Song details timeout."), 5000);
        });
    }

    updateButtons() {
        if (this.playlistManager.isPlaying) {
            $("#play_btn").find('i').removeClass('fa-play');
            $("#play_btn").find('i').addClass('fa-pause');
        } else {
            $("#play_btn").find('i').removeClass('fa-pause');
            $("#play_btn").find('i').addClass('fa-play');
        }
    }

    playSong() {
        this.playlistManager.playSong();
        this.updateButtons();
    }

    pauseSong() {
        this.playlistManager.pauseSong();
        this.updateButtons();
    }

    togglePlay() {
        if (this.playlistManager.isPlaying) {
            this.pauseSong();
        } else {
            this.playSong();
        }
    }

    nextSong(has_ended = false, update_score = true) {
        const songControlPanel = this;
        if (this.currentSongHash != null && update_score) {
            if (has_ended)
                this.socket.emit('emit_music_page_set_song_play_rate', {
                    "hash": songControlPanel.currentSongHash, 
                    "skip_score_change": +1
                });
            else
                this.socket.emit('emit_music_page_set_song_play_rate', {
                    "hash": songControlPanel.currentSongHash, 
                    "skip_score_change": -1
                });
        }

        this.playlistManager.nextSong();
    }

    previousSong() {
        this.playlistManager.previousSong();
    }
}

export default SongControlPanel;