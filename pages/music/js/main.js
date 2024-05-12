// Create a closed scope to avoid any variable collisions  
(function() {
  //// CONSTANTS AND VARIABLES

  //// BEFORE PAGE LOADED
  // Show only the active tab and hide all others
  function showTab(tab_name) { 
    $('.tab-content').each(function() {
      $(this).hide();
    });
  
    $('.tab-button').each(function() {
      $(this).removeClass('is-active')
    })
  
    // Show the selected tab
    $(`#tab_button_${tab_name}`).addClass('is-active');
    $(`#tab_content_${tab_name}`).show()
  }

  //// AFTER PAGE LOADED
  $(document).ready(function() {
    $("#tab_button_library").click(()=>{
      showTab('library')
    })
    
    $("#tab_button_radio").click(()=>{
      showTab('radio')
    })
  })

  //// RESPONDS TO SOCKET EVENTS
})();