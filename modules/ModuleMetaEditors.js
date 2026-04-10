import MetaEditor from '/modules/MetaEditor.js';

/**
 * Creates a pair of MetaEditor instances for a module:
 *   - An editable .meta file editor
 *   - A read-only full-description viewer
 *
 * Returns { openMetaEditor, openFullDescription } functions.
 *
 * @param {object}  socket      – Socket.IO client instance
 * @param {string}  moduleName  – e.g. "images", "music", "videos"
 */
export default function createModuleMetaEditors(socket, moduleName) {
  const prefix = `emit_${moduleName}_page`;

  const metaEditor = new MetaEditor({
    api: {
      load: (filePath, onLoaded) => {
        socket.emit(`${prefix}_get_external_metadata_file_content`, filePath, (response) => {
          onLoaded(response.content || '');
        });
      },
      save: async (filePath, content) => {
        socket.emit(`${prefix}_save_external_metadata_file_content`, {
          file_path: filePath,
          metadata_content: content
        });
      }
    }
  });

  const fullDescriptionEditor = new MetaEditor({
    api: {
      load: (filePath, onLoaded) => {
        socket.emit(`${prefix}_get_full_metadata_description`, filePath, (response) => {
          onLoaded(response.content || '');
        });
      },
      save: () => Promise.resolve()
    },
    readOnly: true,
  });

  function openMetaEditor(fileData) {
    metaEditor.open({
      filePath: fileData.file_path,
      displayName: fileData.base_name || '',
    });
  }

  function openFullDescription(fileData) {
    fullDescriptionEditor.open({
      filePath: fileData.file_path,
      displayName: (fileData.base_name + ' full search description') || '',
    });
  }

  return { openMetaEditor, openFullDescription };
}
