/**
 * Reusable SearchBarComponent:
 * - Renders mode/order/temperature buttons, search input/button, and keyword tags
 * - Keeps state in sync with URL (text_query, mode, order, temperature, seed)
 * - Configurable modes, keywords, and visibility of controls
 */
export default class SearchBarComponent {
  constructor(options = {}) {
    // Defaults
    this.options = {
      container: null,                    // selector or HTMLElement
      enableModes: ['file-name', 'semantic-content', 'semantic-metadata'],
      showOrder: true,
      showTemperature: true,
      temperatures: [0, 1, 5],
      keywords: [],                       // ['recommendation', 'rating', ...]
      autoSyncUrl: true,                  // update URL on search and reload
      ensureDefaultsInUrl: true,          // if params missing on load -> add and reload
      paramNames: {
        text: 'text_query',
        mode: 'mode',
        order: 'order',
        temperature: 'temperature',
        seed: 'seed',
        page: 'page',
      },
      defaults: {
        mode: 'file-name',
        order: 'most-relevant',
        temperature: 0,
        text_query: '',
      },
      onSearch: null,                     // (state) => {}
      onStateChange: null,                // (state) => {}
      // Optional custom labels/icons
      labels: {
        modes: {
          'file-name': { title: 'Filename-based search', content: '🔤' },
          'semantic-content': { title: 'Content-based semantic search', content: '🧠' },
          'semantic-metadata': { title: 'Metadata-based semantic search', content: '📝' },
        },
        orders: {
          'most-relevant': { title: 'Most relevant first', iconClass: 'fas fa-arrow-up-short-wide' },
          'least-relevant': { title: 'Least relevant first', iconClass: 'fas fa-arrow-down-short-wide' },
        },
        temperatures: {
          0: { title: 'Temperature 0 (deterministic results)', iconClass: 'fas fa-thermometer-empty' },
          0.2: { title: 'Temperature 0.2 (slight randomness)', iconClass: 'fas fa-thermometer-quarter' },
          1: { title: 'Temperature 1 (balanced randomness)', iconClass: 'fas fa-thermometer-half' },
          2: { title: 'Temperature 2 (more random results)', iconClass: 'fas fa-thermometer-three-quarters' },
        },
      },
    };

    // Merge options (shallow)
    this.options = { ...this.options, ...options };
    this.options.labels = { ...this.options.labels, ...(options.labels || {}) };
    this.options.labels.modes = { ...this.options.labels.modes, ...((options.labels && options.labels.modes) || {}) };
    this.options.labels.orders = { ...this.options.labels.orders, ...((options.labels && options.labels.orders) || {}) };
    this.options.labels.temperatures = { ...this.options.labels.temperatures, ...((options.labels && options.labels.temperatures) || {}) };

    this.container = typeof this.options.container === 'string'
      ? document.querySelector(this.options.container)
      : this.options.container;

    if (!this.container) {
      throw new Error('SearchBarComponent: container not found');
    }

    // Initial state from URL or defaults
    this.state = this.readStateFromUrl();

    // If required, ensure defaults exist in URL (and reload once)
    if (this.options.ensureDefaultsInUrl) {
      const changed = this.ensureDefaultsInUrl();
      if (changed) return; // page will reload
    }

    // Build UI
    this.root = this.buildUI();

    // Initialize active states in UI
    this.syncUIWithState();

    // Bind events
    this.bindEvents();
  }

  // --- URL/state helpers ---

  readStateFromUrl() {
    const p = new URLSearchParams(window.location.search);
    const { text, mode, order, temperature, seed } = this.options.paramNames;

    return {
      text_query: p.get(text) ? decodeURIComponent(p.get(text)) : this.options.defaults.text_query,
      mode: p.get(mode) || this.options.defaults.mode,
      order: p.get(order) || this.options.defaults.order,
      temperature: p.get(temperature) != null ? parseFloat(p.get(temperature)) : this.options.defaults.temperature,
      seed: p.get(seed) || null,
    };
  }

  ensureDefaultsInUrl() {
    const p = new URLSearchParams(window.location.search);
    const names = this.options.paramNames;
    let changed = false;

    // mode
    if (!p.get(names.mode)) {
      p.set(names.mode, this.state.mode);
      changed = true;
    }
    // order
    if (!p.get(names.order)) {
      p.set(names.order, this.state.order);
      changed = true;
    }
    // temperature
    if (!p.get(names.temperature)) {
      p.set(names.temperature, String(this.state.temperature));
      changed = true;
    }
    // seed
    if (!p.get(names.seed)) {
      const newSeed = Math.floor(Math.random() * 1e9);
      this.state.seed = String(newSeed);
      p.set(names.seed, this.state.seed);
      changed = true;
    }
    if (changed && this.options.autoSyncUrl) {
      // Preserve other params (e.g., page/path)
      window.location.search = p.toString();
      return true;
    }
    return false;
  }

  updateUrlOnSearch() {
    const p = new URLSearchParams(window.location.search);
    const names = this.options.paramNames;

    p.set(names.mode, this.state.mode);
    p.set(names.order, this.state.order);
    p.set(names.temperature, String(this.state.temperature));
    p.set(names.text, encodeURIComponent(this.state.text_query || ''));
    p.set(names.page, '1'); // reset to page 1 on new search

    // Always create a fresh seed for a new search
    this.state.seed = String(Math.floor(Math.random() * 1e9));
    p.set(names.seed, this.state.seed);

    window.location.search = p.toString();
  }

  // --- UI creation ---

  buildUI() {
    // Root container
    const wrapper = document.createElement('div');

    // Top controls line
    const controlsField = document.createElement('div');
    controlsField.className = 'field is-grouped';

    // Mode buttons
    if (this.options.enableModes && this.options.enableModes.length > 0) {
      this.modeGroup = this.createModeButtons(this.options.enableModes);
      const ctrl = document.createElement('div');
      ctrl.className = 'control buttons has-addons';
      ctrl.appendChild(this.modeGroup);
      controlsField.appendChild(ctrl);
    }

    // Order buttons
    if (this.options.showOrder) {
      this.orderGroup = this.createOrderButtons();
      const ctrl = document.createElement('div');
      ctrl.className = 'control buttons has-addons';
      ctrl.appendChild(this.orderGroup);
      controlsField.appendChild(ctrl);
    }

    // Temperature buttons
    if (this.options.showTemperature) {
      this.temperatureGroup = this.createTemperatureButtons(this.options.temperatures);
      const ctrl = document.createElement('div');
      ctrl.className = 'control buttons has-addons';
      ctrl.appendChild(this.temperatureGroup);
      controlsField.appendChild(ctrl);
    }

    // Search input + button
    const searchCtrl = document.createElement('div');
    searchCtrl.className = 'control is-expanded';
    searchCtrl.innerHTML = `
      <div class="field has-addons">
        <div class="control is-expanded">
          <input class="input is-fullwidth" type="text" placeholder="Write your search query or place a path to a music file here.">
        </div>
        <div class="control">
          <button class="button is-info">Search</button>
        </div>
      </div>
    `;
    this.searchInput = searchCtrl.querySelector('input');
    this.searchButton = searchCtrl.querySelector('button');
    controlsField.appendChild(searchCtrl);

    wrapper.appendChild(controlsField);

    // Keywords row
    if (this.options.keywords && this.options.keywords.length > 0) {
      const keywordsField = document.createElement('div');
      keywordsField.className = 'field is-grouped is-grouped-multiline mt-4';
      const label = document.createElement('b');
      label.textContent = 'Special keywords:';
      keywordsField.appendChild(label);

      this.keywordButtons = [];
      for (const kw of this.options.keywords) {
        const btn = document.createElement('button');
        btn.className = 'tag is-info is-light';
        btn.textContent = kw;
        btn.type = 'button';
        btn.addEventListener('click', () => {
          this.searchInput.value = kw;
          this.state.text_query = kw;
          this.triggerSearch();
        });
        this.keywordButtons.push(btn);
        keywordsField.appendChild(btn);
      }
      wrapper.appendChild(keywordsField);
    }

    // Mount
    this.container.innerHTML = '';
    this.container.appendChild(wrapper);
    return wrapper;
  }

  createModeButtons(modes) {
    const frag = document.createDocumentFragment();
    modes.forEach((m, idx) => {
      const meta = this.options.labels.modes[m] || { title: m, content: m };
      const btn = document.createElement('button');
      btn.className = 'button search-mode-btn';
      btn.dataset.mode = m;
      btn.title = meta.title;
      btn.innerHTML = `<span>${meta.content}</span>`;
      if (m === this.state.mode) btn.classList.add('is-active');
      btn.addEventListener('click', () => {
        this.setMode(m);
      });
      frag.appendChild(btn);
      if (idx === 0 && !this.state.mode) this.state.mode = m;
    });
    const container = document.createElement('div');
    container.appendChild(frag);
    return container;
  }

  createOrderButtons() {
    const container = document.createElement('div');

    const mkBtn = (order, label) => {
      const meta = this.options.labels.orders[order] || { title: order, iconClass: '' };
      const btn = document.createElement('button');
      btn.className = 'button search-order-btn';
      btn.dataset.order = order;
      btn.title = meta.title;
      btn.innerHTML = `<span class="icon"><i class="${meta.iconClass}"></i></span>`;
      if (order === this.state.order) btn.classList.add('is-active');
      btn.addEventListener('click', () => {
        this.setOrder(order);
      });
      return btn;
    };

    container.appendChild(mkBtn('most-relevant'));
    container.appendChild(mkBtn('least-relevant'));
    return container;
  }

  createTemperatureButtons(temps) {
    const container = document.createElement('div');
    temps.forEach((t) => {
      const meta = this.options.labels.temperatures[t] || { title: `Temperature ${t}`, iconClass: '' };
      const btn = document.createElement('button');
      btn.className = 'button temperature-btn';
      btn.dataset.temperature = String(t);
      btn.title = meta.title;
      btn.setAttribute('aria-label', `Temperature ${t}`);
      btn.innerHTML = `<span class="icon is-medium"><i class="${meta.iconClass}"></i></span>`;
      if (Number(t) === Number(this.state.temperature)) btn.classList.add('is-active');
      btn.addEventListener('click', () => {
        this.setTemperature(t);
      });
      container.appendChild(btn);
    });
    return container;
  }

  // --- Event binding ---

  bindEvents() {
    this.searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        this.state.text_query = this.searchInput.value;
        this.triggerSearch();
      }
    });

    this.searchButton.addEventListener('click', () => {
      this.state.text_query = this.searchInput.value;
      this.triggerSearch();
    });
  }

  // --- State <-> UI sync ---

  syncUIWithState() {
    // Input
    if (this.searchInput) {
      this.searchInput.value = this.state.text_query || '';
    }
    // Modes
    if (this.modeGroup) {
      const buttons = this.modeGroup.querySelectorAll('.search-mode-btn');
      buttons.forEach(b => {
        b.classList.toggle('is-active', b.dataset.mode === this.state.mode);
      });
    }
    // Order
    if (this.orderGroup) {
      const buttons = this.orderGroup.querySelectorAll('.search-order-btn');
      buttons.forEach(b => {
        b.classList.toggle('is-active', b.dataset.order === this.state.order);
      });
    }
    // Temperature
    if (this.temperatureGroup) {
      const buttons = this.temperatureGroup.querySelectorAll('.temperature-btn');
      buttons.forEach(b => {
        const btnTemp = parseFloat(b.dataset.temperature);
        b.classList.toggle('is-active', btnTemp === Number(this.state.temperature));
      });
    }
  }

  // --- Public API ---

  getState() {
    return { ...this.state };
  }

  getSelectedMode() {
    return this.state.mode;
  }

  getSelectedOrder() {
    return this.state.order;
  }

  getSelectedTemperature() {
    return this.state.temperature;
  }

  getTextQuery() {
    return this.state.text_query || '';
  }

  setMode(mode) {
    if (!this.options.enableModes.includes(mode)) return;
    // Preserve current input value
    if (this.searchInput) {
      this.state.text_query = this.searchInput.value;
    }
    this.state.mode = mode;
    this.syncUIWithState();
    this.notifyStateChange();
  }

  setOrder(order) {
    if (!['most-relevant', 'least-relevant'].includes(order)) return;
    // Preserve current input value
    if (this.searchInput) {
      this.state.text_query = this.searchInput.value;
    }
    this.state.order = order;
    this.syncUIWithState();
    this.notifyStateChange();
  }

  setTemperature(t) {
    const value = Number(t);
    if (!this.options.temperatures.map(Number).includes(value)) return;
    // Preserve current input value
    if (this.searchInput) {
      this.state.text_query = this.searchInput.value;
    }
    this.state.temperature = value;
    this.syncUIWithState();
    this.notifyStateChange();
  }

  setTextQuery(text) {
    this.state.text_query = text || '';
    this.syncUIWithState();
    this.notifyStateChange();
  }

  focusInput() {
    this.searchInput?.focus();
  }

  triggerSearch() {
    if (typeof this.options.onSearch === 'function') {
      this.options.onSearch(this.getState());
    }
    if (this.options.autoSyncUrl) {
      this.updateUrlOnSearch();
    }
  }

  notifyStateChange() {
    if (typeof this.options.onStateChange === 'function') {
      this.options.onStateChange(this.getState());
    }
  }
}