# Change History


## TODO list
### News 
Create some ideas for theme based news maps. 
Think over how to add the current time to the prompt and take it into account when adding news to the database. This should add to the LLM the concept of the relevance of a particular event.
Predict tags of the news and allow to filter the news by this tags.
 
### Music 
Add progress bar when music library updating.    
Add a way to change rest of metadata of the song.
Add volume control.
Allow user to click at any song to play it.
Add button that allows to start or continue radio session.
Add message if there is new unindexed media files. 
Add "What would you like to listen today?" text field and start session button.
Add a way to activate or deactivate AIDJ.

### Fine-tuning 
Make the progress bar active when the model is fine-tuned.  
Add a way to set "Number of training epochs”, "Maximum token size" and "OpenAssistant dataset percent used" before starting fine-tuning.  
Disable the start button if fine-tuning has already started.  
Implement "self-aware" fine tuning mode to make it possible to chat with the project about itself.

### Search
Draft some ideas of how it can automatically search for lyrics of the music and add them to the music files after the user approval.

### Wiki
Display current wiki structure.
Display current active wiki page similarly to how the main page is presented.

### LLM engine
Add an ability to send text to the front-end token by token.

### General
Add an ability to load out models from the GPU by pressing the button.  
Move to single page architecture or find any other way to store the current state of the media player. Possibly using vue.js.  
Add some sort of a control panel that is active on all pages (may be hidden) and shows current GPU memory load by the system and console output.  
Add some way of displaying active page at the head menu.  
Replace all "georgesung/llama2_7b_chat_uncensored" instances with info from config.yaml.  
Find some way of adding some sort of "extensions" where we can see data from some arbitrary service such as twitter, youtube, reddit and so on and rates this data locally.  
Create an extension that provides recent Arxiv papers (https://arxiv.org/list/cs.LG/recent).  
Make a queue for LLM engine processing.  
Remove all dependencies on CDN services.

## Versions History

### Version 0.0.5 (14.01.2024)
Added LLM loading indication.  
Added tables markdown extension for proper wiki rendering.  
TTS initialization changed to on demand approach for faster start up time.  
Added wiki pages rendering. The 'wiki' button has been removed, as everything is acceptable through the main page. 
Audio elements removed from base.html and add them to music.html.
Added message about TTS initialization when AIDJ been activated.
Added music library navigation system.
Added icon change on play button when song is paused. 
Radio related code has been moved to separate js file, also some code cleanup. 
Now radio history is not lost when music page has been reloaded.

### Version 0.0.4 (06.01.2024)
Added installation info to the README.md file.  
Fixed some bugs that cause full_play_count and skip_count not being updated.  
Fixed bug that caused page to scroll up when closing song edit modal window.  
Fixed a dramatic mistake in fine-tunning, where text based data were sampled line by line instead of document based sampling as it was intended.  
Added "Self-aware" mode to the fine-tunning, that adds main code of the project to the LLM training data. 
Better representation of radio mode play history.  
Tabs on the music page are responsive now.  
Now TextPredictor is a singleton class, that should eliminate problems with copying LLM in several processing.  
Added "The Gardian" news aggregator.  

### Version 0.0.3 (16.11.2023)
Radio mode converted to a state based approach to see everything that is happening at the backend in a real time and stack prepared announcements and songs in a buffer.  
Images of song covers now send from backend to frontend as base64 images.  
There is now multiple prompts for AI DJ. It should help to avoid repetitions.  
Added a edit button that allows to change song metadata from the frontend UI. Right now only the lyrics of the song is actually changes.  

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