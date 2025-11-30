### **Anagnorisis: Search Your Data Effectively (v0.3.0)**

### Opening
Welcome to the ‘Anagnorisis’. ‘Anagnorisis’ is a local recommendation system designed to let you search, filter, and enjoy your personal data securely, all on your local machine. Anagnorisis could learn from *your* feedback to provide a truly personalized experience. Unlike centralized services that control what you see and collect your personal information, Anagnorisis learns from your feedback to provide a truly personalized experience completely locally on your device only.

The goal is to give you powerful tools for information management, where the algorithms would work for you and not the other way around. 

Today I am gonna present you the new powerful search capabilities introduced in new version of the project.

### Unified Search Bar
One of the most significant improvements in this version is the unified search bar that is now available in all modules. Let's look at how it is structured.

First you can see one of the three search modes available: `file-name-based` search, `semantic-content-based` search and `semantic-metadata-based` search.
Next to that, you have priority control, which lets you find either the most or least relevant results for your query.
Then there's the temperature control, which allows you to adjust the randomness of the search results. This is a fantastic tool for discovery, and we’ll take a closer look at it in just a bit.
Of course, you have the search bar itself, where you can type your query and press 'Enter' or click the 'Search' button.
And finally, you have a list of special keywords. These are powerful, module-specific filters that let you sort your data in unique ways.

### Search Modes
Let's see how each of these search modes works in practice.

#### File-name-based search
`File-name-based` search is your classic, fast fuzzy search. When this mode is active, you can quickly find files by their names or the names of the folders they're contained in. You don't have to be too precise here, as the search is tolerant to typos. This is perfect for when you know exactly what you're looking for.

This search mode works exactly the same in all modules of the system.

*(Show examples: searching for "track_04.mp3" in Music, "vacation photos" in Images, or "project draft" in Text, "guitar lesson video" in Video modules.)*

#### Semantic-content-based search
`Semantic-content-based` search is a more advanced search mode that uses modern embedding models to understand the actual content of your files. This search mode is especially useful when you don't yet know much about your data, do not have any notes attached to it and want to find something solely based on the content inside the files.

This search mode is already available in the Music, Images and Text modules. There no such mode in the Video module yet, as video content analysis usually a more complex task that requires more resources. Meanwhile this project tries to be as friendly to a modern home user's hardware as possible. *??The video content-based search mode might be added in the future versions if a suitable lightweight solution would be found.??*

*(Show examples: searching for "Relaxing instrumental music" in Music, "A cat sitting on a windowsill" in Images, or "Research papers about machine learning" in Text.)*


#### Semantic-metadata-based search
`Semantic-metadata-based` search is another advanced search mode that uses embedding models to understand the metadata associated with your files. This search mode analyzes file names, folder paths, internal file's metadata and any additional notes you have stored in external `.meta` files. This search mode is especially useful when you have already organized your data with meaningful file and folder names or added your own custom descriptions to your files, as it uses that context to find the most relevant results.

Let me show you a powerful, concrete example. Here’s a photo of an old man. Inside Anagnorisis, I can easily add a description to it by pressing at "Edit .meta file" at the context menu. 
Let's simply add the note, "My grandfather."

*(Visual cue: Show the UI for editing metadata and typing "My grandfather".)*

This action creates a simple text file right next to my image, with the same name but with a `.meta` extension. You can edit this file directly with any text editor, giving you full and easy control over your external metadata that do not affect the original file in any way.

*(Visual cue: Switch to a file explorer view, showing the image file and the new `.meta` file next to it. Briefly open the .meta file in a simple text editor.)*

Now, watch what happens If I search for "my grandpa," the image appears. But because Anagnorisis uses text embeddings to understand the *meaning* behind the words, the search is largely language-independent. You can search in any language you like *(showing "мой дедушка"...  "a nagypapam"...)* and the result are going to be roughly the same. This allows you to build a rich, searchable context for your data in any language you're comfortable with, with any words you have in mind at the moment. There are no special formatting rules required; just write naturally and the search engine will try to understand your intention when looking for the data you have requested.

## Controlling Your Results

Finding the right content is only half the story. Anagnorisis also gives you precise control over how your results are presented.

### Priority Control

By default, results are sorted by "most relevant," bringing the best matches to the top. But what if you’re looking for something you haven't seen in a while, or something that's the opposite of your query? Switching the priority to "least relevant" turns the search on its head, helping you find outliers or simply rediscover forgotten corners of your library.

### Temperature Control

The temperature setting is your control knob for randomness and discovery. At zero, you get a strictly ordered list based on relevance — perfect for finding a specific item.

As you increase the temperature, you introduce a degree of controlled randomness. This is your tool for discovery, helping you break out of filter bubbles and rediscover forgotten gems within your own collection. It intelligently shuffles the results so that highly relevant items are still likely to appear fist, but less obvious ones also get a chance to surface.

This mode works especially well in the music module, as any of your search results could be immediately transformed into a playlist for listening session. So when you are looking for something very specific you can find it by the name with `file-name-based` search mode, description of the mood of the song with `semantic-content-based` *(calm music for relaxation)* search mode or any notes you have attached to the music file with `semantic-metadata-based` search mode. But you can also do that, while setting the temperature to a higher value to get a more varied playlist that is still based on your query.

### Special Keyword Filtering
Now let's see how the special keywords work.

When you click the special keyword in the search bar the filtering will perform some special operation on your data. Each module has its own set of special keywords that are relevant to the type of data it works with. These search results will not be affected by the search mode selected, but yet still could be controlled with the priority and temperature settings.

### Special Keyword Filtering

Now let's look at the special keywords. When you click on a keyword, it performs a special filtering operation on your data. Each module has its own set of special keywords that are relevant to the current type of the data it is working with.

For example, in the **Music** module, you can filter by `similarity` among the tracks, `length` or `file size` of the tracks, completely randomly, by your personal `rating`, or even get a `recommendation` from your locally trained model recommendation engine. You can see how the `recommendation` engine works in the internal wiki page of the project.

In the **Images** module, you can filter by `similarity`, `resolution`, `proportion`, or `file size` to find exactly the kind of image you need.

?? Text

?? Video

### Find Similar Files

Finally, you can always click at any file in the search results and find a similar files to it by pressing the `Find similar` button. This will perform similarity-based search using the selected file as a query.

### Why This Matters
Every feature we've just seen is built on a foundation of transparency and user control. All of this processing, learning, and searching happens securely on your local machine. Your data, your ratings, and your habits are never sent to a third party. It’s about building a deeply personal relationship with your own data, transforming it from a static archive into a dynamic, explorable part of your digital life.

### Outro
To get started and explore these features yourself, visit the Anagnorisis GitHub repository, where you can download the project, read the documentation, and stay updated on the progress. 

Thank you for watching. Till the next time.

Snips:
- Content-based semantic search is useful when you don't yet know much about your data, do not have any notes attached to it and want to find something solely based on the content inside the files.
- Content-based search example "Music for meditation".
- The **temperature** setting allows you to control the randomness of your search results. At zero, you get a strictly ordered list based on relevance—perfect for finding a specific item." "As you increase the temperature, you introduce a degree of randomness. This is your tool for discovery, helping you break out of filter bubbles and rediscover forgotten gems within your own library."
- Every feature in Anagnorisis is built on a foundation of transparency and user control. Unlike the black-box algorithms of centralized platforms, these tools are open for you to see and command. When you search, you choose the mode. When you want discovery, you adjust the temperature. When you rate a song, you are directly training *your* personal evaluation model, making the system more attuned to your unique tastes over time.
- "This is what it means to control the algorithm, not the other way around. It’s about building a deeply personal relationship with your own data, transforming it from a static archive into a dynamic, explorable part of your digital life."
- Anagnorisis is designed from the ground up to be a completely local and private way to search, filter, and enjoy your personal data. Unlike centralized services that control what you see and collect your personal information, Anagnorisis runs entirely on your machine. The goal is to give you a powerful tool for information management, where the algorithms work for you, not the other way around.
- All of this processing, learning, and searching happens securely on your local machine. Your data, your ratings, and your habits are never sent to a third party. You have complete ownership and control. Anagnorisis is about empowering you to manage your digital life effectively and privately while providing search features previously available only through (big corp?) centralized services.
- We've built these tools to be transparent, so you understand how your results are being generated, and powerful, so you can find exactly what you're looking for in your ever-growing library of personal data.