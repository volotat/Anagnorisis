The project has the support to view and edit both internal and external metadata for media files, enhancing user control over their digital content.
Both types of metadata has their own benefits and downfalls, and the choice between them depends on user needs and preferences.


.meta metadata approach:
    Pros:
        - Non-destructive: Does not alter the original media file, preserving its integrity and crucially file's hash that plays and important role in the internal workings of system.
        - Portability: Metadata can be easily transferred between systems without modifying the media files.
        - Compatibility: Avoids potential compatibility issues with software that may not support embedded metadata.
        - Unbounded Size: External files can store larger amounts of metadata without impacting the media file size.
        - Easy to read and edit: Users can easily open and modify .meta files with standard text editors, no extra libraries or tools are needed.
        - .meta file could be dedicated to the whole folder, reducing clutter when many media files share the same metadata.
    Cons:
        - Separation: Metadata is stored separately from the media file, which can lead to loss of metadata if the .meta file is misplaced or deleted.
        - Management: Requires additional management to ensure .meta files are kept in sync with their corresponding media files.
        - Accessibility: Some applications may not recognize or utilize external metadata files.

Embedded metadata approach:
    Pros:
        - Integration: Metadata is stored within the media file itself, ensuring it is always available and reducing the risk of loss.
        - Simplicity: Users do not need to manage separate metadata files, simplifying the workflow.
    Cons:
        - Destructive: Modifying the media file to include metadata can alter its original state and hash.
        - Portability: Moving or copying the media file may not preserve the metadata if not handled properly.
        - Compatibility: Some software may not support or recognize embedded metadata formats.
        - Size Limitations: Embedded metadata may be limited in size depending on the media format.

The choice between using .meta files or embedded metadata should be based on the specific requirements of the user, such as the need for non-destructive editing, ease of management, and compatibility with other software. The project supports both methods, allowing users to select the approach that best fits their workflow and preferences. However, .meta files are the preferred option due to their non-destructive nature and ease of management within the system.