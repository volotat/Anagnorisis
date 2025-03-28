import FolderViewComponent from '/pages/FolderViewComponent.js'; 
import FileGridComponent from '/pages/FileGridComponent.js';
import PaginationComponent from '/pages/PaginationComponent.js';

//// CONSTANTS AND VARIABLES
let num_files_on_page = 60;
let num_files_in_row = 6; // TODO: calculate from the screen size
let selected_files = [];
let currentFilePath = null; // To store currently opened file path in modal

//// BEFORE PAGE LOADED

// read page number from the URL ?page=1
const urlParams = new URLSearchParams(window.location.search);
let pageParam = parseInt(urlParams.get('page'));
let page = (!pageParam || pageParam < 1) ? 1 : pageParam;
let text_query = urlParams.get('text_query') || '';
let path = urlParams.get('path') || '';
text_query = decodeURIComponent(text_query);
path = decodeURIComponent(path);
console.log('path', path);

function replaceNewLinesWithBreaks(text) {
    return text.replace(/\n/g, '<br>');
}

function renderTextPreview(fileData) { // Function for Text module preview
    const previewText = document.createElement('p');
    previewText.className = 'text-preview'; // Add class for styling (optional)
    previewText.textContent = fileData.preview_text || 'No preview available';
    previewText.style.overflow = 'hidden'; // basic text styling for preview
    previewText.style.textOverflow = 'ellipsis';
    //previewText.style.whiteSpace = 'nowrap';
    previewText.style.maxWidth = '100%';
    previewText.style.wordBreak = 'break-word';
    previewText.style.textAlign = 'justify';
    previewText.style.whiteSpace = 'break-spaces';
    return previewText;
}

function renderCustomData(fileData) { // Function for custom data rendering
    const dataContainer = document.createElement('div');
    dataContainer.className = 'file-custom-data';

    // File Path
    const filePathElement = document.createElement('p');
    filePathElement.className = 'file-info file-path';
    filePathElement.innerHTML = `<b>Path:</b>&nbsp;${fileData.file_path}`;
    dataContainer.appendChild(filePathElement);

    // Hash
    const hashElement = document.createElement('p');
    hashElement.className = 'file-info file-hash';
    hashElement.innerHTML = `<b>Hash:</b>&nbsp;${fileData.hash}`;
    dataContainer.appendChild(hashElement);

    // User Rating
    const userRatingElement = document.createElement('p');
    userRatingElement.className = 'file-info file-user-rating';
    userRatingElement.innerHTML = `<b>User rating:</b>&nbsp;${fileData.user_rating !== null ? fileData.user_rating : 'N/A'}`;
    dataContainer.appendChild(userRatingElement);

    // Model Rating
    const modelRatingElement = document.createElement('p');
    modelRatingElement.className = 'file-info file-model-rating';
    modelRatingElement.innerHTML = `<b>Model rating:</b>&nbsp;${fileData.model_rating !== null ? fileData.model_rating : 'N/A'}`;
    dataContainer.appendChild(modelRatingElement);

    // File Size
    const fileSizeElement = document.createElement('p');
    fileSizeElement.className = 'file-info file-size';
    fileSizeElement.innerHTML = `<b>Size:</b>&nbsp;${fileData.file_size}`;
    dataContainer.appendChild(fileSizeElement);

    return dataContainer;
}


//// AFTER PAGE LOADED
$(document).ready(function() {
    // Request current media folder path
    socket.emit('emit_text_page_get_path_to_media_folder');

    // Request folders for folder view
    socket.emit('emit_text_page_get_folders', {
        path: path, 
    }); 

    // Request files from the main media folder
    socket.emit('emit_text_page_get_files', {
      path: path, 
      pagination: (page-1)*num_files_on_page, 
      limit: page * num_files_on_page,
      text_query: text_query 
    });

    let currentFilePath = null; // To store currently opened file path

    // --- Folder View ---
    socket.on('emit_text_page_show_folders', (data) => { // NEW event handler for folders
        const folderView = new FolderViewComponent(data.folders, data.folder_path);
        $('.menu').html(folderView.getDOMElement());
    });

    // --- File List ---
    let paginationComponent; // Declare paginationComponent in the scope 
    socket.on('emit_text_page_show_files', (data) => {
        console.log('emit_text_page_show_files', data);

        // Update or Initialize FileGridComponent
        const fileGridComponent = new FileGridComponent({
            containerId: '#text_files_grid_container', // Use the new container ID
            filesData: data.files_data,
            renderPreviewContent: renderTextPreview, // Pass the text preview function
            renderCustomData: renderCustomData, // No custom data rendering for now
            handleFileClick: (fileData) => {
                currentFilePath = fileData.file_path; // Store current file path
                socket.emit('emit_text_page_get_file_content', { file_path: fileData.file_path });
                $('#text_file_modal_title').text(fileData.base_name); // Set modal title to filename
                $('#text_file_modal').addClass('is-active'); // Show modal
                $('#text_content_textarea').val('Loading...'); // Loading text
                // Switch to Raw Text tab initially
                showTab('raw');
            },
            numColumns: num_files_in_row, 
        });

        // Update or Initialize Pagination Component
        const paginationContainer = $('.pagination.is-rounded.level-left.mb-0 .pagination-list'); // Select pagination container
        const urlParams = new URLSearchParams(window.location.search); // Get URL parameters for pattern
        let urlPattern = `?page={page}`; // Base URL pattern

        if (urlParams.get('text_query')) { // Add text_query if present
            urlPattern += `&text_query=${encodeURIComponent(urlParams.get('text_query'))}`;
        }

        if (!paginationComponent) { // Instantiate PaginationComponent if it doesn't exist yet
            paginationComponent = new PaginationComponent({
                containerId: paginationContainer.closest('.pagination').get(0), // Pass the pagination nav element
                currentPage: page,
                totalPages: Math.ceil(data["total_files"] / num_files_on_page),
                urlPattern: urlPattern,
            });
        } else { // Update existing PaginationComponent
            paginationComponent.updatePagination(page, Math.ceil(data["total_files"] / num_images_on_page));
        }
    });

    // Function to switch tabs in the modal
    function showTab(tabName) {
        $('.tab-content').hide(); // Hide all tab contents
        $('.tabs li').removeClass('is-active'); // Deactivate all tabs
        $(`.tabs li[data-tab="${tabName}"]`).addClass('is-active'); // Activate current tab
        $(`#text_viewer_${tabName}`).show(); // Show current tab content
    }
    
 
    // Tab switching logic
    $('.tabs li').on('click', function() {
        const tabName = $(this).data('tab');
        showTab(tabName);
    });

    // Close modal when clicking modal-close elements
    $('.modal-close-action').on('click', function() {
        $('#text_file_modal').removeClass('is-active'); // Hide modal
    });
  
    // Save button in modal (initially just for raw text)
    $('#text_file_modal .modal-save-action').on('click', function() {
        const textContent = $('#text_content_textarea').val(); // Get content from textarea
        socket.emit('emit_text_page_save_file_content', { file_path: currentFilePath, text_content: textContent }); // Use currentFilePath
        $('#text_file_modal').removeClass('is-active'); // Hide modal after save
    });


    // Show file content in the modal
    socket.on('emit_text_page_show_file_content', (data) => {
        const file_path = data.file_path;
        const text_content = data.content;
        $('#text_content_textarea').val(text_content); // Set text content in textarea           
    });

    // Display current search status
    socket.on('emit_show_search_status', (status) => {
        $('.search-status').html(status);
    });
});