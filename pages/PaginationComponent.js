class PaginationComponent {
    constructor({ containerId, currentPage, totalPages, urlPattern }) {
      this.containerId = containerId;
      this.currentPage = currentPage;
      this.totalPages = totalPages;
      this.urlPattern = urlPattern; // URL pattern with placeholders for page and text_query
      this.render();
    }
  
    render() {
      const container = $(this.containerId);
      container.empty(); // Clear any existing pagination
  
      if (this.totalPages <= 1) {
        return; // No pagination needed if only one page or less
      }
  
      const paginationList = $('<ul></ul>').addClass('pagination-list');
  
      // Ellipsis for start and end
      const ellipsis = '<li><span class="pagination-ellipsis">&hellip;</span></li>';
  
      // Add "Previous" button (if not on the first page) - removed for now to simplify, can be added back if needed
      /*if (this.currentPage > 1) {
        const prevPageLink = this.createPageLink(this.currentPage - 1, 'Previous');
        paginationList.append(prevPageLink);
      }*/
  
      // Add page links
      for (let i = 1; i <= this.totalPages; i++) {
        if (i === 1 || i === this.totalPages || Math.abs(i - this.currentPage) <= 2) {
          const pageLink = this.createPageLink(i, i.toString());
          if (i === this.currentPage) {
            pageLink.find('a').addClass('is-current'); // Mark current page as active
          }
          paginationList.append(pageLink);
        } else if ((i === 2 && this.currentPage > 4) || (i === this.totalPages - 1 && this.currentPage < this.totalPages - 3)) {
          paginationList.append(ellipsis);
        }
      }
  
      // Add "Next" button (if not on the last page) - removed for now to simplify, can be added back if needed
      /*if (this.currentPage < this.totalPages) {
        const nextPageLink = this.createPageLink(this.currentPage + 1, 'Next');
        paginationList.append(nextPageLink);
      }*/
  
      container.append(paginationList);
    }
  
    createPageLink(pageNumber, text) {
      const url = this.generatePageUrl(pageNumber);
      const listItem = $('<li></li>');
      const link = $('<a></a>')
        .addClass('pagination-link')
        .attr('href', url)
        .attr('aria-label', `Goto page ${pageNumber}`)
        .text(text);
      listItem.append(link);
      return listItem;
    }
  
    // generatePageUrl(pageNumber) {
    //   // Replace placeholders in urlPattern with actual values
    //   let url = this.urlPattern.replace('{page}', pageNumber);
    //   return url;
    // }

    generatePageUrl(pageNumber) {
      // Check if URL already has a page parameter (either page=number or page={page})
      if (this.urlPattern.includes('page=')) {
        // Replace existing page parameter with new page number
        // This regex matches both page=123 and page={page}
        return this.urlPattern.replace(/page=(?:\d+|\{page\})/, `page=${pageNumber}`);
      } else {
        // Add page parameter to URL
        const separator = this.urlPattern.includes('?') ? '&' : '?';
        return `${this.urlPattern}${separator}page=${pageNumber}`;
      }
    }
  
    updatePagination(currentPage, totalPages) {
      this.currentPage = currentPage;
      this.totalPages = totalPages;
      this.render(); // Re-render with updated page numbers
    }
  }
  
  export default PaginationComponent;