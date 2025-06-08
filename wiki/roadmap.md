# Roadmap beyond v0.1.0 update
## Music Module Overhaul
Align the music module with recent file management and caching improvements. Implement batch operations, advanced filtering, and streamlined embedding storage similar to the Images module, enhancing performance and user control. Fix DB structure.

Status: Completed.

## Text recommendation module
Develop a text recommendation module to complement the existing image and music modules. Implement text embedding, search, and recommendation features, expanding the platform’s capabilities to textual data. Implement highlighting of the most relevant parts of the text in the document.

## Project refactoring and reorganization
Refactor the codebase to improve organization and maintainability. Also, as a result the text search from the file names+metadata should be possible in all modules with some ability to perform text-based search/sematic search or both depending on the user needs at the moment.

## Video recommendation module
Introduce a new module for video data, enabling efficient embedding, search, and recommendation capabilities. Implement UI features for video search, rating, and filtering to enhance user experience and platform functionality.

## Data Server implementation and integration
Split the data sharing and management functionality into a separate Data Server module or service. This should allow to users to create their own Data Server and share it with others, allowing to share data across multiple devices and users. Sharing local data should also be performed through the Data Server, although in a completely automatic fashion without user intervention. Users should be able to share their data for localhost, local network, or globally as needed. Single user should be able to have multiple Data Servers active at the same time with different levels of permissions and sharing capabilities. 
This is going to be a first step towards a p2p media sharing platform, where the recommendation engine works locally on the user's machine and completely independently from the data providers.

## Universal Embedding Model
Research and develop a unified embedding model that generalizes across data types (images, music, videos, text). It should store embeddings in float array where values are sorted by their importance through specialized training procedure. This should allow to perform efficient search and recommendation across huge amount of data as it will only require to compare arrays up to selected K values to get an estimate of similarity. This will allow users to share their embeddings with others to provide a fast search and recommendation system on p2p basis.

## Unified Evaluation Model
Develop a single evaluation model that is reliant on Universal Embedding Model. This should create a cohesive evaluation system strictly based on the particular user's feedback and represent the user's preferences in a monolithic way.

## Optimizations and Performance Improvements
Experiment with various optimization techniques to enhance the platform’s performance and scalability such as FAISS integration, caching strategies, and parallel processing. Implement performance monitoring tools to identify bottlenecks and optimize resource utilization for better user experience.

