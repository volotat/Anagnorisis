# Change History


## TODO list
### News 
Create some ideas for theme based news maps. 

### Music 
Add progress bar when music library updating.  
Add icon change on play button when song is paused.    
Add music control to the common page header.  
Display music structure in a way as it is stored in the folders.
Add a way to change metadata of the song.
Add volume control.
Convert radio mode to a state based approach to see everything that is happening at the backend in a real time and stuck prepared announcements and songs in a stuck.

### Fine-tuning 
Make the progress bar active when the model is fine-tuned.  
Add a way to set "Number of training epochs”, "Maximum token size" and "OpenAssistant dataset percent used" before starting fine-tuning.  
Disable the start button if fine-tuning has already started.  

### Wiki
Display current wiki structure.
Display current active wiki page similarly to how the main page is presented.

### LLM engine
Add an ability to send text to the front-end token by token.

### General
Add an ability to load out models from the GPU by pressing the button.  
Move to single page architecture or find any other way to store the current state of the media player.  
Add some sort of a control panel that is active on all pages (may be hidden) and shows current GPU memory load by the system and console output. 
Add installation info to the readme.md file. 

## Versions History
### Version 0.0.2 (10.11.2023)
Added skip multiplier to music recommendation algorithm.  
Added display of the music cover to the music page.  
Added internal playback history and switching to the previous track.  
Added play button icon change when paused.  
Now you can start playing random song after the page music loaded by pressing the play button.  
Now you can change current song play time by clicking on the progress bar.  
Added initial version of AI DJ mode.  
Added 'config.yaml' that contains all currently available settings of the application. 

### Initial version 0.0.1 (01.09.2023)
Initial sever and frontend architecture implemented.  
Basic news rating using LLAMA 2 implemented.  
Basic music player with user's ratings implemented.  
Front page of the server displays the README.md file.  