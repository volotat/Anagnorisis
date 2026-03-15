export default class ContextMenuComponent {
  constructor() {
    this.container = document.createElement('div');
    this.container.id = 'file-grid-context-menu';
    this.container.className = 'dropdown';
    this.container.style.position = 'absolute';
    this.container.style.zIndex = '1000';
    this.container.style.display = 'none';
    this.container.innerHTML = `
      <div class="dropdown-menu" role="menu" style="display:block;">
        <div class="dropdown-content"></div>
      </div>
    `;
    document.body.appendChild(this.container);

    // Close on outside click or Escape
    document.addEventListener('click', (e) => {
      if (!this.container.contains(e.target)) this.hide();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.hide();
    });
  }

  show(x, y, items = []) {
    const content = this.container.querySelector('.dropdown-content');
    content.innerHTML = '';

    for (const item of items) {
      if (item.type === 'divider') {
        const hr = document.createElement('hr');
        hr.className = 'dropdown-divider';
        content.appendChild(hr);
        continue;
      }

      const a = document.createElement('a');
      a.href = '#';
      a.className = 'dropdown-item';
      a.innerHTML = `
        ${item.icon ? `<span class="icon is-small mr-2"><i class="${item.icon}"></i></span>` : ''}
        <span>${item.label}</span>
      `;
      a.addEventListener('click', (e) => {
        e.preventDefault();
        this.hide();
        if (typeof item.action === 'function') item.action();
      });
      content.appendChild(a);
    }

    this.container.style.left = `${x}px`;
    this.container.style.top = `${y}px`;
    this.container.style.display = 'block';
    this.container.classList.add('is-active');
  }

  hide() {
    this.container.classList.remove('is-active');
    this.container.style.display = 'none';
  }
}