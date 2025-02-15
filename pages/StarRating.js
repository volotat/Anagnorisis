class StarRatingComponent {
  constructor({ initialRating = null, callback = ()=>{} }) {
    this.maxRating = 10;
    this.rating = initialRating;
    this.containers = [];
    this.callback = callback;
    this.isUserRated = false;
  }

  issueNewHtmlComponent(params) {
    const ratingContainer = new StarRatingHTMLContainer(this, params);
    this.containers.push(ratingContainer);
    return ratingContainer.container;
  }

  updateAllContainers() {
    this.containers.forEach(container => container.updateDisplay());
  }
}

class StarRatingHTMLContainer {
  constructor(starRatingObject, { containerType = 'div', size = 34, isActive = false, showPassiveAsNumber = true } = {}) {
    this.starRatingObject = starRatingObject;
    this.isActive = isActive;
    this.showPassiveAsNumber = showPassiveAsNumber;
    this.container = document.createElement(containerType);

    // Set cursor to pointer so the mouse changes to a select state over the whole container.
    this.container.style.cursor = 'pointer';

    // Numeric indicator element
    this.indicator = document.createElement('span');
    this.container.appendChild(this.indicator);
    
    // Add event listeners to the indicator for 0 rating
    if (isActive) {
      this.indicator.addEventListener('mousemove', (e) => {
        this.updateDisplay(0);
      });
      this.indicator.addEventListener('click', (e) => {
        this.starRatingObject.rating = 0;
        this.starRatingObject.callback(0);
        this.starRatingObject.updateAllContainers();
      });
      this.indicator.addEventListener('mouseleave', (e) => {
        this.updateDisplay();
      });
    }

    // Container for stars
    this.starsWrapper = document.createElement('span');
    this.container.appendChild(this.starsWrapper);

    // Basic styling
    this.container.style.userSelect = 'none';
    this.starsWrapper.style.display = 'inline-block';

    // Create star elements (noninteractive here)
    this.generateStarDisplay(size);
    this.updateDisplay();

    // Only one set of event handlers on the parent container:
    if (this.isActive) {
      this.starsWrapper.addEventListener('mousemove', (e) => {
        const rect = this.starsWrapper.getBoundingClientRect();
        // Compute overall rating = fraction across total width times maxRating
        let fractionOverall = (e.clientX - rect.left) / rect.width;
        fractionOverall = Math.min(Math.max(fractionOverall, 0), 1);
        const previewRating = fractionOverall * this.starRatingObject.maxRating;
        this.updateDisplay(previewRating);
      });
      this.starsWrapper.addEventListener('click', (e) => {
        const rect = this.starsWrapper.getBoundingClientRect();
        let fractionOverall = (e.clientX - rect.left) / rect.width;
        fractionOverall = Math.min(Math.max(fractionOverall, 0), 1);
        const newRating = fractionOverall * this.starRatingObject.maxRating;
        this.starRatingObject.rating = newRating;
        this.starRatingObject.callback(newRating);
        this.starRatingObject.updateAllContainers();
        this.starRatingObject.isUserRated = true;
      });
      this.starsWrapper.addEventListener('mouseleave', () => {
        this.updateDisplay();
      });
    }
  }

  generateStarDisplay(size) {
    this.starElements = [];
    for (let i = 1; i <= this.starRatingObject.maxRating; i++) {
      const starContainer = document.createElement('span');
      starContainer.classList.add('star-container');
      starContainer.style.position = 'relative';
      starContainer.style.display = 'inline-block';
      starContainer.style.fontSize = size+'px';
      starContainer.style.lineHeight = '1';
      starContainer.style.margin = '0 1px';

      // Empty star layer
      const emptyStar = document.createElement('span');
      emptyStar.classList.add('star-empty');
      emptyStar.textContent = '☆';

      // Filled star overlay
      const filledStar = document.createElement('span');
      filledStar.classList.add('star-filled');
      filledStar.textContent = '★';
      filledStar.style.position = 'absolute';
      filledStar.style.top = '0';
      filledStar.style.left = '0';
      filledStar.style.overflow = 'hidden';
      filledStar.style.width = '0%';
      filledStar.style.pointerEvents = 'none';

      starContainer.appendChild(emptyStar);
      starContainer.appendChild(filledStar);
      this.starsWrapper.appendChild(starContainer);
      this.starElements.push(starContainer);
    }
  }

  updateDisplay(previewRating = null) {
    // Use previewRating if provided, else the actual rating
    let rating = previewRating == null ? this.starRatingObject.rating : previewRating;

    // For noninteractive mode, show numeric text only.
    if (!this.isActive && this.showPassiveAsNumber) {
      this.container.innerHTML =
        rating == null
          ? 'Not rated yet'
          : rating.toFixed(1) + '/' + this.starRatingObject.maxRating.toString();
      return;
    }

    let color_class = this.starRatingObject.isUserRated ? 'has-text-link' : 'has-text-info';
    this.container.classList.remove('has-text-link', 'has-text-info');
    this.container.classList.add(color_class);

    // Optionally update an indicator. Here we simply use it to show a bullet if set.
    this.indicator.textContent = rating === null ? '◦' : '•';

    // Update each star fill based on its index.
    for (let j = 0; j < this.starElements.length; j++) {
      const starContainer = this.starElements[j];
      const filledStar = starContainer.querySelector('.star-filled');
      let starValue = rating - j;
      if (starValue < 0) starValue = 0;
      if (starValue > 1) starValue = 1;
      filledStar.style.width = (starValue * 100).toFixed(0) + '%';
    }
  }
}


export default StarRatingComponent;