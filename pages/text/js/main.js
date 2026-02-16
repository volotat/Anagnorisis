import FolderViewComponent from '/pages/FolderViewComponent.js'; 
import FileGridComponent from '/pages/FileGridComponent.js';
import PaginationComponent from '/pages/PaginationComponent.js';
import SearchBarComponent from '/pages/SearchBarComponent.js';
import StarRatingComponent from '/pages/StarRating.js';

//// CONSTANTS AND VARIABLES
let num_files_on_page = 24;
let num_files_in_row = 6; // TODO: calculate from the screen size
let selected_files = [];
let currentFilePath = null; // To store currently opened file path in modal

//// BEFORE PAGE LOADED

// read page number from the URL ?page=1
const urlParams = new URLSearchParams(window.location.search);
let pageParam = parseInt(urlParams.get('page'));
let page = (!pageParam || pageParam < 1) ? 1 : pageParam;
// let text_query = urlParams.get('text_query') || '';
let path = urlParams.get('path') || '';
// text_query = decodeURIComponent(text_query);
path = decodeURIComponent(path);
console.log('path', path);

/**
 * Shortens a string by keeping a specified number of characters 
 * at the start and end, replacing the middle with '...'.
 * @param {string} str The string to shorten.
 * @param {number} charsToShow The number of characters to show at the start and end.
 * @returns {string} The shortened string.
 */
function shortenHash(str, charsToShow = 8) {
    if (!str || str.length <= charsToShow * 2) {
        return str;
    }
    const start = str.substring(0, charsToShow);
    const end = str.substring(str.length - charsToShow);
    return `${start}...${end}`;
}

function replaceNewLinesWithBreaks(text) {
    return text.replace(/\n/g, '<br>');
}

function renderTextPreview(fileData) { // Function for Text module preview
    const previewText = document.createElement('p');
    previewText.className = 'text-preview'; // Add class for styling
    previewText.textContent = fileData.file_info.preview_text || 'No preview available';
    
    // Set basic styling
    previewText.style.overflow = 'hidden';
    previewText.style.textOverflow = 'ellipsis';
    previewText.style.maxWidth = '100%';
    previewText.style.wordBreak = 'break-word';
    previewText.style.whiteSpace = 'break-spaces';
    previewText.style.fontFamily = "'Courier New', monospace";
    
    // Set line height in em or px
    const lineHeight = 1.5; // 1.5em line height
    previewText.style.lineHeight = `${lineHeight}em`;
    
    // Set max height based on number of lines
    const maxLines = 12; // Show maximum 10 lines of text
    previewText.style.maxHeight = `${maxLines * lineHeight}em`;
    
    return previewText;
}

function renderCustomData(fileData) { // Function for custom data rendering
    const dataContainer = document.createElement('div');

    // Search matching scores
    if (fileData.search_score !== null && fileData.search_score !== undefined) {
        const searchScoresElement = document.createElement('p');
        searchScoresElement.className = 'file-info file-search-scores';
        searchScoresElement.innerHTML = `<b>Search Score:</b>&nbsp;${(fileData.search_score || 0).toFixed(3)}`;
        dataContainer.appendChild(searchScoresElement);
    }

    const data = document.createElement('p');
    data.style.wordBreak = 'break-all';

    $(data).append('<b>Path:</b>&nbsp;' + fileData.file_path + '<br>');
    $(data).append('<b>Hash:</b>&nbsp;' + shortenHash(fileData.hash) + '<br>');

    // User Rating
    if (fileData.file_info.user_rating != null) {
        $(data).append('<b>User rating:</b>&nbsp;' + fileData.file_info.user_rating.toFixed(2) + '/10<br>');
    } else {
        $(data).append('<b>User rating:</b>&nbsp;N/A<br>');
    }

    // Model Rating
    if (fileData.file_info.model_rating != null) {
        $(data).append('<b>Model rating:</b>&nbsp;' + fileData.file_info.model_rating.toFixed(2) + '/10<br>');
    } else {
        $(data).append('<b>Model rating:</b>&nbsp;N/A<br>');
    }

    $(data).append('<b>File size:</b>&nbsp;' + fileData.file_size + '<br>');

    dataContainer.appendChild(data);

    return dataContainer;
}

function renderMarkdown(textContent) {
    try {
        // Configure marked to treat single line breaks as <br>
        marked.setOptions({
            breaks: true,  // Enable GitHub-flavored line breaks
            gfm: true      // Enable GitHub Flavored Markdown
        });
        
        let htmlContent = marked.parse(textContent);
        return DOMPurify.sanitize(htmlContent);
    } catch (error) {
        console.error('Error parsing markdown:', error);
        return `<p>Error rendering markdown: ${error.message}</p>`;
    }
}

function renderHTML(textContent) {
    // For security, sanitize HTML content using DOMPurify to prevent XSS
    return DOMPurify.sanitize(textContent);
}

function updateTabContent(textContent) {
    // Raw text tab
    $('#text_content_textarea').val(textContent);
    
    // Markdown tab
    const markdownHtml = renderMarkdown(textContent);
    $('#text_content_markdown').html(markdownHtml);
    
    // HTML tab  
    const htmlContent = renderHTML(textContent);
    $('#text_content_html').html(htmlContent);
}

function setupAutoUpdate() {
    const textarea = $('#text_content_textarea');
    
    // Function to update rendered content
    function updateRenderedContent() {
        const currentContent = textarea.val();
        
        // Update markdown tab
        const markdownHtml = renderMarkdown(currentContent);
        $('#text_content_markdown').html(markdownHtml);
        
        // Update HTML tab  
        const htmlContent = renderHTML(currentContent);
        $('#text_content_html').html(htmlContent);
    }
    
    // Real-time updates on input
    textarea.on('input keyup paste', function() {
        // Use requestAnimationFrame for smooth updates
        requestAnimationFrame(updateRenderedContent);
    });
}


//// AFTER PAGE LOADED
$(document).ready(function() {
    let paginationComponent;

    // Instantiate SearchBarComponent
    const searchBar = new SearchBarComponent({
        container: '#search_bar_container',
        enableModes: ['file-name', 'semantic-content'], // disable here as needed
        showOrder: true,
        showTemperature: true,
        temperatures: [0, 0.2, 1, 2],
        keywords: [], //['recommendation', 'rating', 'random', 'file_size', 'length', 'similarity'],
        autoSyncUrl: true,
        ensureDefaultsInUrl: true,
    });

    const search_state = searchBar.getState();

    // Request current media folder path
    socket.emit('emit_text_page_get_path_to_media_folder');

    // Request files from the main media folder
    // --- Folder View ---
    socket.emit('emit_text_page_get_folders', {
            path: path, 
        }, (response) => { // event handler for folders
            console.log('emit_text_page_get_folders', response);

            const folderView = new FolderViewComponent(response.folders, response.folder_path, true); // Enable context menu
            $('.menu').append(folderView.getDOMElement());
        }
    );

    // --- File List ---
    let currentFilePath = null; // To store currently opened file path
    socket.emit('emit_text_page_get_files', {
      path: path, 
      pagination: (page-1)*num_files_on_page, 
      limit: num_files_on_page,
      text_query: search_state.text_query,
      seed: search_state.seed,
      mode: search_state.mode,
      order: search_state.order,
      temperature: search_state.temperature,
    }, (response) => {
        console.log('emit_text_page_show_files', response);

        // Update or Initialize FileGridComponent
        const fileGridComponent = new FileGridComponent({
            containerId: '#text_files_grid_container', // Use the new container ID
            filesData: response.files_data,
            renderPreviewContent: renderTextPreview, // Pass the text preview function
            renderCustomData: renderCustomData, // No custom data rendering for now
            handleFileClick: (fileData) => {
                currentFilePath = fileData.file_path; // Store current file path
                socket.emit('emit_text_page_get_file_content', { file_path: fileData.file_path });
                $('#text_file_modal_title .scrolling-title').text(fileData.base_name); // Set modal title to filename

                // Create an interactive star rating component for the modal
                const modalRatingContainer = document.getElementById('text_file_modal_rating');
                modalRatingContainer.innerHTML = ''; // Clear previous rating widget
                
                // Determine which rating to show: user rating if available, otherwise model rating
                const hasUserRating = fileData.file_info.user_rating !== null && fileData.file_info.user_rating !== undefined;
                const displayRating = hasUserRating ? fileData.file_info.user_rating : fileData.file_info.model_rating;
                
                const modalStarRating = new StarRatingComponent({
                    callback: (rating) => {
                        socket.emit('emit_text_page_set_text_rating', {
                            hash: fileData.hash,
                            file_path: fileData.file_path,
                            rating: rating,
                        });
                        // Update in-memory data so reopening the modal shows the new rating
                        fileData.file_info.user_rating = rating;
                        // Mark as user-rated and update display colors
                        modalStarRating.isUserRated = true;
                        modalStarRating.updateAllContainers();
                    },
                    initialRating: displayRating,
                });
                
                // Override isUserRated flag if showing model rating
                modalStarRating.isUserRated = hasUserRating;
                
                const starRatingElement = modalStarRating.issueNewHtmlComponent({
                    containerType: 'span',
                    isActive: true,
                });
                modalRatingContainer.appendChild(starRatingElement);

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
        const urlPattern = `?${urlParams.toString()}`;

        if (!paginationComponent) { // Instantiate PaginationComponent if it doesn't exist yet
            paginationComponent = new PaginationComponent({
                containerId: paginationContainer.closest('.pagination').get(0), // Pass the pagination nav element
                currentPage: page,
                totalPages: Math.ceil(response["total_files"] / num_files_on_page),
                urlPattern: urlPattern,
            });
        } else { // Update existing PaginationComponent
            paginationComponent.updatePagination(page, Math.ceil(response["total_files"] / num_files_on_page));
        }
    });

    

    // // --- Folder View ---
    // socket.on('emit_text_page_show_folders', (data) => { // NEW event handler for folders
    //     const folderView = new FolderViewComponent(data.folders, data.folder_path, true); // Enable context menu
    //     $('.menu').append(folderView.getDOMElement());
    // });

    
    // let paginationComponent; // Declare paginationComponent in the scope 
    // socket.on('emit_text_page_show_files', (data) => {
        
    // });

    // Function to switch tabs in the modal
    function showTab(tabName) {
        $('.tab-content').hide(); // Hide all tab contents
        $('.tabs li').removeClass('is-active'); // Deactivate all tabs
        $(`.tabs li[data-tab="${tabName}"]`).addClass('is-active'); // Activate current tab
        
        if (tabName === 'raw') {
            $('#text_viewer_raw').show();
            $('#text_viewer_markdown').hide();
            $('#text_viewer_html').hide();
        } else if (tabName === 'markdown') {
            $('#text_viewer_raw').hide();
            $('#text_viewer_markdown').show();
            $('#text_viewer_html').hide();
        } else if (tabName === 'html') {
            $('#text_viewer_raw').hide(); 
            $('#text_viewer_markdown').hide();
            $('#text_viewer_html').show();
        }
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
        // $('#text_content_textarea').val(text_content); // Set text content in textarea   
        
        // Update all tab contents
        updateTabContent(text_content);

        // Setup auto-update functionality
        setupAutoUpdate();
    });

    // Display current search status
    socket.on('emit_show_search_status', (status) => {
        $('.search-status').html(status);
    });

    // Show current media folder path
    socket.on('emit_text_page_show_path_to_media_folder', (current_path) => {
        $('#path_to_media_folder').val(current_path);
    });

    // Update path to the media folder
    $('#update_path_to_media_folder').click(function() {
        socket.emit('emit_text_page_update_path_to_media_folder', $('#path_to_media_folder').val());
        // refresh page after a short delay
        setTimeout(function() {
            location.reload();
        }, 500);
    });
});