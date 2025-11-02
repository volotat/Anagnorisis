### **Anagnorisis: Search Your Data Effectively (v0.3.0)**

## V1.0 Script

**(Intro Music with project logo animation)**

**Host:** Welcome back to Anagnorisis, the open-source system that puts you in control of your personal data. For anyone new here, Anagnorisis is a completely local recommendation system. It allows you to search, filter, and enjoy your private media—like music, photos, and documents—securely on your own computer, with an AI that learns directly from your feedback.

The philosophy behind Anagnorisis is simple: your data is a reflection of you, and you should be the one to control it. We're building an alternative to cloud-based services, giving you the tools to manage your digital life privately and effectively, free from external algorithms or data breaches.

**(Cut to a sleek screen recording of the Anagnorisis UI, showcasing the unified search bar across different modules)**

**Host:** In version 0.3.0, we've taken a massive leap forward in making data discovery more powerful and intuitive. We've unified the search experience across all modules, introducing advanced filtering and new ways to explore your collections. Let’s dive into what's new.

**(Focus on the search bar, highlighting the different modes: "File Name," "Semantic," and the new "Metadata" search)**

**Host:** The heart of this update is the enhanced search bar. We've expanded on our powerful semantic search capabilities. You can still search by **file name** for quick lookups or use **semantic search** to find content based on meaning, like "relaxing instrumental music" or "photos from my beach vacation."

But now, we've introduced a dedicated **metadata search**. This allows you to query the specific details within your files. For images, this could be camera settings or resolution. For music, it could be the album, artist, or even the bitrate. And with our new `.meta` file integration, you can add your own tags and notes to any file and make them instantly searchable.

**(Demonstrate a search in the Images module. The user types a query, and the results appear, with a new filter panel visible)**

**Host:** Let's look at the Images module. Finding the right photo is now easier than ever. We've added a powerful BRISQUE score filter, which allows you to find images based on objective quality assessment, helping you quickly sift through and find your best shots. We are also working on a more effective resolution estimation, so you can find images that are perfect for any purpose without a lengthy manual search.

**(Transition to the Music module. Show a user activating the new "Chain Mode")**

**Host:** The Music module has received some exciting updates as well. Introducing **"Chain Mode."** When activated, Anagnorisis will create a seamless playlist where each song is selected based on its similarity to the one that just played. It’s a fantastic way to explore sonic landscapes within your library and discover connections between your favorite tracks.

We've also added more granular controls, including a dedicated volume control and the ability to edit metadata for any song right from the library view, making it easier than ever to keep your collection organized.

**(Introduce the new "Deep Research" concept with a futuristic UI mockup)**

**Host:** But the most forward-looking feature we're introducing is the foundation for our upcoming **"Deep Research" module.** Imagine a tool that uses your personalized evaluation models to search through vast datasets, like research papers from Arxiv, and highlights the information that is most relevant to *your* specific interests.

This new functionality will leverage a user-trained text model to sift through information and recommend articles, documents, and other text-based content that aligns with your unique preferences. It’s a step towards turning Anagnorisis into not just a media manager, but a true personal information assistant.

**(Show a final montage of quick, effective searches across different data types, emphasizing speed and relevance)**

**Host:** All of these enhancements are powered by significant backend improvements, including the integration of the FAISS library for lightning-fast vector searches and more optimized storage for embeddings. This means that even as your libraries grow, Anagnorisis will remain fast and responsive.

**(Return to the host)**

**Host:** Anagnorisis is an active, community-driven project. Our goal is to create a powerful, private, and transparent tool for information management. If this vision resonates with you, check out our GitHub repository linked in the description. You can download the project, see our full change history, and even contribute to its development.

**(Outro music and end screen with a prominent link to the GitHub page)**

**Host:** Thank you for joining us for this look at version 0.3.0. Take control of your data, and we'll see you in the next update.

## V2.0 Script

### Anagnorisis: Search your data effectively (v0.3.0) - Video Script

**Video Length:** ~3-4 minutes
**Style:** Informative, calm, and slightly philosophical, consistent with previous materials.
**Visuals:** Screen recordings of the Anagnorisis UI, focusing on the search bar in the Music and Images modules. Simple text overlays to highlight key terms.

---

**(0:00 - 0:25) [Intro Music - calm, ambient electronic track]**

**Visual:** A slow zoom on the Anagnorisis logo, followed by quick cuts showing different modules (Music, Images, Text) being populated with files.

**Narrator (V.O.):**
"For years, we've entrusted our digital lives to centralized services. Our music, our photos, our thoughts—all filtered through algorithms we can't see or control. These systems often prioritize engagement or promotion over our personal satisfaction. What if we could take back control?

This is the vision behind Anagnorisis: a completely local, private, and open-source recommendation system. It’s designed to help you search, filter, and enjoy your personal data securely, right on your own machine, with an AI that learns directly from your feedback."

---

**(0:26 - 1:30) [Scene: The Unified Search Bar]**

**Visual:** The screen shows the 'Images' module. The mouse cursor highlights the new, unified search bar at the top of the screen.

**Narrator (V.O.):**
"Today, we're taking a deep dive into the heart of the Anagnorisis experience: a powerful, unified search engine that puts you in command. It’s more than just a search bar; it's a precise tool for exploring your data exactly how you want."

**Visual:** The user types "a cat sitting on a windowsill" into the search bar. The mode is set to "Semantic Content". A grid of relevant cat photos from the local library appears.

**Narrator (V.O.):**
"The search engine operates in three distinct modes. The first is **Semantic Content** search. Here, the system uses AI-powered embedding models to understand the *meaning* behind your query. You don't need to remember a file name. You can simply describe what you're looking for, and Anagnorisis will find the most relevant images, music, or text from your library."

**Visual:** Switch to the 'Music' module. The user types "upbeat instrumental rock" into the search bar, still in "Semantic Content" mode. A list of matching music files appears.

**Narrator (V.O.):**
"Looking for 'happy music' or 'songs for a rainy day'? The system intelligently analyzes the content of your music files to provide suggestions that match the mood."

**Visual:** The user changes the search mode to "File Name". They type "track_04.mp3". The search results instantly filter to that specific file. The `rapidfuzz` library is briefly mentioned in a text overlay.

**Narrator (V.O.):**
"The second mode is **File Name** search. Powered by `rapidfuzz`, this provides lightning-fast and accurate results when you know exactly what you're looking for."

**Visual:** The user switches the mode to "Semantic Metadata". They type "Vacation Photos". The results show images whose file paths or associated `.meta` files contain those keywords. The score breakdown (total, content, meta) is visible for each result.

**Narrator (V.O.):**
"Finally, there's **Semantic Metadata** search. This powerful mode looks beyond the content and searches through file names, folder paths, and even custom data you've stored in external `.meta` files. This allows for incredibly intuitive organization, letting you find all files from a specific project or event just by describing it."

---

**(1:31 - 2:20) [Scene: Advanced Search Controls]**

**Visual:** The cursor now points to the "temperature" control next to the search bar. The user performs a search for "relaxing music" with the temperature at 0. The results are strictly ordered by relevance.

**Narrator (V.O.):**
"But true control means going beyond just finding files. Anagnorisis gives you tools to shape your discovery process. The **temperature** setting allows you to control the randomness of your search results. At zero, you get a strictly ordered list based on relevance—perfect for finding a specific item."

**Visual:** The user slides the temperature control higher. The search results re-order, bringing some less-obvious but still relevant tracks towards the top.

**Narrator (V.O.):**
"As you increase the temperature, you introduce a degree of randomness. This is your tool for discovery, helping you break out of filter bubbles and rediscover forgotten gems within your own library."

**Visual:** The user performs another search, this time for images using special keywords. They type `resolution > 1920x1080` and the grid updates. Then they type `similarity: [path/to/image.jpg]` and visually similar images are shown.

**Narrator (V.O.):**
"The search bar also supports special keywords for advanced filtering. You can find files by `file_size`, image `resolution`, or even find tracks of a specific `length`. The `similarity` keyword helps you find duplicates or visually similar images, while `rating` prioritizes music based on your personal scores."

---

**(2:21 - 3:05) [Scene: Connecting to the Core Vision]**

**Visual:** A montage of the different search modes and filters being used quickly. The screen shows a user rating a song, then retraining the model on the 'Train' page.

**Narrator (V.O.):**
"Every feature in Anagnorisis is built on a foundation of transparency and user control. Unlike the black-box algorithms of centralized platforms, these tools are open for you to see and command. When you search, you choose the mode. When you want discovery, you adjust the temperature. When you rate a song, you are directly training *your* personal evaluation model, making the system more attuned to your unique tastes over time."

**Visual:** The screen returns to a clean view of the main dashboard. The camera slowly pans across the UI.

**Narrator (V.O.):**
"This is what it means to control the algorithm, not the other way around. It’s about building a deeply personal relationship with your own data, transforming it from a static archive into a dynamic, explorable part of your digital life."

---

**(3:06 - 3:30) [Outro]**

**Visual:** The Anagnorisis logo appears on screen with the GitHub URL below it (github.com/volotat/Anagnorisis).

**Narrator (V.O.):**
"Anagnorisis is an open-source project under active development, with new features and improvements always on the horizon.

Visit the GitHub repository to download the project, explore the code, and stay updated on the latest enhancements. Thank you for watching."

**[Outro Music fades in and plays to the end]**

## V3.0 Script

Anagnorisis: Search your data effectively (v0.3.0) - Video Script

**(Intro Music with project logo: Anagnorisis)**

**Host:** Hello and welcome back to Anagnorisis, the open-source recommendation system that puts you in control of your personal data.

**(Cut to a clean, minimalist graphic showing icons for music, photos, and documents flowing from a cloud icon to a local computer icon with the Anagnorisis logo on it.)**

**Host:** For anyone new to the project, Anagnorisis is designed from the ground up to be a completely local and private way to search, filter, and enjoy your personal data. Unlike centralized services that control what you see and collect your personal information, Anagnorisis runs entirely on your machine. The goal is to give you a powerful tool for information management, where the algorithms work for you, not the other way around.

**Host:** Today, we're going to take a deep dive into the powerful and unified search system that has been refined in the latest versions. We've made finding the perfect file, whether it's a song or an image, more intuitive and transparent than ever.

**(Transition to a screen recording of the Anagnorisis Music Module interface.)**

**Host:** Let's start with the Music module. The first thing you'll notice is the new, advanced search bar. This component is now shared across all modules for a consistent experience. It allows for much more than just simple keyword matching.

**(Host types "upbeat rock music" into the search bar and hits enter. The library view populates with relevant music.)**

**Host:** This is a semantic search. Anagnorisis isn't just looking for filenames; it understands the *meaning* behind your query. It analyzes the audio itself to find music that matches the vibe you're looking for.

**Host:** But we can get much more specific. The search bar now has different modes. In 'file-name' mode, it uses a fast search to find tracks by their title or path.

**(Host switches the mode to "file-name" and types a part of a song title. The results are filtered instantly.)**

**Host:** There's also 'semantic-metadata' mode, which intelligently searches through your file and folder names. This is great for finding all songs from a specific "live recordings" folder, for example.

**(The screen shows the search results, with a small tooltip displaying score breakdowns: semantic, metadata, total.)**

**Host:** And to maintain full transparency, you can see exactly *why* a song was recommended, with a clear breakdown of its content and metadata scores.

**Host:** Beyond the text search, you can use special keywords for powerful filtering.

**(The host demonstrates the following searches with quick cuts.)**

**Host:** Looking for high-quality audio? Try `file_size:>50MB`. Want a song that's longer than 10 minutes? Use `length:>10m`.

**Host:** You can also ask for your personal favorites with `rating:>=4`, or get a truly personalized playlist with the `recommendation` keyword. This uses Anagnorisis's unique recommendation engine, which considers your ratings, how often you skip a track, and when you last heard it, to create a fresh and engaging list just for you.

**(Host points to the "temperature" slider in the search bar.)**

**Host:** And for those moments of discovery, you can adjust the 'temperature.' A low temperature gives you the most precise, relevant results. As you increase it, you introduce more randomness, helping you rediscover forgotten gems in your collection.

**(Transition to the Images Module interface.)**

**Host:** Now, let's see how these powerful features translate to your image library. The same advanced search bar is right here.

**(Host types "a photo of a sunset over the ocean" into the search bar. The grid fills with sunset pictures.)**

**Host:** Just like with music, this is a full semantic search. The system analyzes the *content* of your images to find what you're looking for.

**Host:** We also have our special filters. `proportion:tall` is perfect for finding a new phone wallpaper.

**(The grid updates to show only vertically oriented images.)**

**Host:** You can find visually similar images by selecting a photo and using the `similarity` filter. This is incredibly useful for finding duplicates or other photos from the same event.

**(A picture of a cat is selected, and the similarity search brings up other photos of the same cat.)**

**Host:** And of course, filters like `resolution`, `file_size`, and `rating` give you precise control to find exactly what you need, whether it's your highest-rated photos or just very large files you might want to clean up.

**(Cut back to the host.)**

**Host:** All of this processing, learning, and searching happens securely on your local machine. Your data, your ratings, and your habits are never sent to a third party. You have complete ownership and control. Anagnorisis is about empowering you to manage your digital life effectively and privately.

**Host:** We've built these tools to be transparent, so you understand how your results are being generated, and powerful, so you can find exactly what you're looking for in your ever-growing library of personal data.

**(Outro music starts, with the GitHub link displayed on screen: github.com/volotat/Anagnorisis)**

**Host:** To get started and explore these features yourself, visit the Anagnorisis GitHub repository. You can download the project, read the documentation, and stay updated on our progress. Thank you for watching, and see you next time.