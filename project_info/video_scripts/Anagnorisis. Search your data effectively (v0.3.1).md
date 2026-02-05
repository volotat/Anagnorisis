### **Anagnorisis: Search Your Data Effectively (v0.3.1)**

### Opening
Welcome to the ‘Anagnorisis’. ‘Anagnorisis’ is a local recommendation system designed to let you search, filter, and enjoy your personal data securely, all on your local machine. Unlike centralized services that control what you see and collect your personal information, Anagnorisis learns from your feedback completely locally on your device only.

The goal is to give you powerful tool for information management, where the algorithm would work for you and not the other way around. 

Today I am gonna present you the new powerful search capabilities introduced in new version of the project.

### Unified Search Bar
One of the most significant improvements in this version is the unified search bar that is now available in all modules. Let's look at how it is structured.

First you can see one of the three search modes available: `file-name-based` search, `content-based` search and `metadata-based` search.
Next to that, you have priority control, which lets you find either the most or least relevant results for your query.
Then there's the temperature control, which allows you to adjust the randomness of the search results. This is a fantastic tool for discovery, and we’ll take a closer look at it in just a bit.
Of course, you have the search bar itself.
And finally, you have a list of special keywords. These are module-specific filters that let you sort your data in unique ways.

### Search Modes
Let's see how each of these search modes works in practice.

#### File-name-based search
`File-name-based` search is your classic, fast fuzzy search. When this mode is active, you can quickly find files by their names or the names of the folders they're contained in. You don't have to be too precise here, as the search is tolerant to typos. This is perfect for when you know exactly what you're looking for.

This search mode works exactly the same in all modules of the system.

*(Show examples: searching for "guitar" in Images showing 'test_guitar.png' file appearing first in the list.)*

For example, if we search for "guitar" in the Images module, with this mode being active, the file named `test_guitar.png` appears first in the results.

#### content-based search
`content-based` search is a more advanced search mode that uses modern embedding models to understand the actual content of your files. This search mode is especially useful when you don't yet know much about your data, do not have any notes attached to it and want to find something solely based on the content inside the files.

This search mode is already available in the Music, Images and Text modules.

*(Show visual cue: searching for "guitar" in Images showing different results based on content)*

Now, if we do the same search for "guitar" in the Images module using `content-based` search mode, we get a variety of images that contain guitars, even if the file names don't include the word "guitar" in it.

*(Show examples: searching for "Relaxing instrumental music" in Music, "A cat sitting on a windowsill" in Images, or "Research papers about machine learning" in Text.)*

The same goes for other modules. In the Music module, searching for "Relaxing instrumental music" brings up tracks that fit that description the most. 

*(Show examples: searching for "machine learning" in Text.)*
In the Text module, searching for "machine learning" retrieves relevant documents based on their content.


#### metadata-based search
`metadata-based` search is another advanced search mode that uses embedding models to understand the metadata associated with your files. This search mode analyzes file names, folder paths, internal file's metadata and any additional notes you have stored in external `.meta` files. This search mode is especially useful when you have already organized your data with meaningful file and folder names or added your own custom descriptions to your files, as it uses that context to find the most relevant results.

For example, if your photos contains information of the place where an image was taken in metadata fields, you can search for these images by simply typing the location in the search bar.

*(Visual cue: Show search for 'photo made in Lisbon' to find images from that place)*

To see the whole description that is used for the metadata-based search, press the "Show full search description" in the context menu.

To add a custom description, select "Edit .meta file" from the context menu. 

*(Visual cue: Show the ubuntu ui with files and .meta files)*

Let me show you how powerful it is on a concrete example. Here’s a photo of some old person. Inside Anagnorisis, I can easily add a description to it, let's simply add the note, "My grandfather."

*(Visual cue: Show the UI for editing metadata and typing "My grandfather".)*

This action creates a simple text file right next to the original image. You can edit this file directly with any text editor, giving you full and easy control over your external metadata that do not affect the original file in any way.

*(Visual cue: Switch to a file explorer view, showing the image file and the new `.meta` file next to it. Briefly open the .meta file in a simple text editor.)*

Now, watch what happens If I search for "my grandpa" using `metadata-based` search.  *(showing search results)* Because Anagnorisis uses text embeddings to understand the *meaning* behind the words, the search is largely language-independent. You can search in any language you like *(showing "мой дедушка"...  "a nagyapám"...)* and the result are going to be roughly the same. This allows you to build a rich, searchable context for your data in any language you're comfortable with, with any words you have in mind at the moment. There are no special formatting rules required; just write naturally and the search engine will try to understand your intention when looking for the data you have requested.

## Controlling Your Results

Finding the right content is only half the story. Anagnorisis also gives you precise control over how your results are presented.

### Priority Control

By default, results are sorted by "most relevant," bringing the best matches to the top. But what if you’re looking for something that's the opposite of your query? Switching the priority to "least relevant" will help you with that.

### Temperature Control

The temperature setting is your control knob for randomness and discovery. At zero, you get a strictly ordered list based on relevance — perfect for finding a specific item.

As you increase the temperature, you introduce a degree of controlled randomness. It intelligently shuffles the results so that highly relevant items are still likely to appear fist, but less obvious ones also get a chance to surface.

This mode works especially well in the music module, as any of your search results could be immediately transformed into a playlist for listening session. So when you are looking for something very specific you can find it by the name, with the description of the mood or the request to notes you have attached to the files. But you can also do that, while setting the temperature to a higher value to get a more varied playlist that is still based on your query.

### Special Keyword Filtering
Now let's see how the special keywords work.

When you click the special keyword in the search bar the filtering will be applied to your data. Each module has its own set of special keywords that are relevant to the type of data it works with. These search results will not be affected by the search mode selected, but yet still could be controlled with the priority and temperature settings.

For example, in the **Music** module, you can filter by `recommendations` from your locally trained recommendation engine, your personal `ratings`, completely randomly, by semantic `similarity` among the tracks, `length` or `file size`. You can see how the `recommendation` engine works in the internal wiki page of the system.

You have the similar filtering options in the **Images** module. Other modules have their own relevant keywords as well.

### Find Similar Files

Finally, you can always click at any file in the search results and find a similar files to it by pressing the `Find similar` button in the context menu. This will perform similarity-based search using the selected file as a query.

### Outro
The project allow you to transform your personal static archive into a dynamic, explorable part of your digital life.

To get started and explore these features yourself, visit the Anagnorisis GitHub repository, where you can download the project, read the documentation, and stay updated on the progress. 

Thank you for watching. Till the next time.