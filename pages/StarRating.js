class StarRatingComponent {
  constructor({ initialRating = null, callback = ()=>{} }) {
    this.maxRating = 10;
    this.rating = initialRating;
    this.containers = [];
    this.callback = callback;
  }

  issueNewHtmlComponent(params) {
    const starRatingHTMLContainer = new StarRatingHTMLContainer(this, params);
    this.containers.push(starRatingHTMLContainer);

    return starRatingHTMLContainer.container;
  }

  updateAllContainers() {
    this.containers.forEach(container => container.updateDisplay());
  }
}

class StarRatingHTMLContainer {
  constructor(starRatingObject, {containerType = 'div', size = 3, isActive = false, showPassiveAsNumber = true}) {
    this.starRatingObject = starRatingObject;
    this.isActive = isActive;
    this.showPassiveAsNumber = showPassiveAsNumber;
    this.symbolsList = [];
    this.container = document.createElement(containerType);
    
    this.container.classList.add(`is-size-${size.toString()}`);
    this.container.classList.add('is-gapless');
    this.container.classList.add('has-text-centered');
    this.container.classList.add('is-unselectable');
  
    this.updateDisplay();
  }

  generateStarDisplay() {
    const starRatingObject = this.starRatingObject;

    // Add the initial symbol based on the rating
    const initialSymbol = document.createElement('span');
    initialSymbol.textContent = starRatingObject.rating === null ? '◦' : '•';
    this.container.appendChild(initialSymbol);
    this.symbolsList.push(initialSymbol);
  
    // Create each star element
    for (let i = 1; i <= starRatingObject.maxRating; i++) {
      const star = document.createElement('span');
      star.textContent = i <= starRatingObject.rating ? '★' : '☆';
      star.classList.add('star');
  
      this.container.appendChild(star);
      this.symbolsList.push(star);
    }

    if (this.isActive) {
      for (let i = 0; i < this.symbolsList.length; i++) {
        this.symbolsList[i].classList.add('is-clickable');

        this.symbolsList[i].addEventListener('mouseover', () => {
          this.updateDisplay(i); 
        });

        this.symbolsList[i].addEventListener('mouseout', () => {
          this.updateDisplay(); 
        });

        this.symbolsList[i].addEventListener('click', () => {
          starRatingObject.rating = i;
          starRatingObject.callback(i);
          starRatingObject.updateAllContainers();
        });
      }
    }
  }

  updateDisplay(tmpRating = null) {
    let rating = this.starRatingObject.rating;
    const maxRating = this.starRatingObject.maxRating;

    if (tmpRating != null) rating = tmpRating;

    if (!this.isActive && this.showPassiveAsNumber) {
      if (rating == null)
        this.container.innerHTML = 'Not rated yet';
      else
        this.container.innerHTML = rating.toString() + '/' + maxRating.toString();

      // Clear the symbols list in case it was previously active for some reason
      this.symbolsList = [];
    } else {
      if (this.symbolsList.length == 0) {
        this.generateStarDisplay();
      }

      this.symbolsList[0].textContent = rating === null ? '◦' : '•';
      for (let j = 1; j <= maxRating; j++) {
        this.symbolsList[j].textContent = j <= rating ? '★' : '☆';
      }
    }
  }
}

export default StarRatingComponent;