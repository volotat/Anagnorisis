// Minimal .meta editor modal (raw text only). No previews or external deps.
// Requires an API with load(filePath, onLoaded) and save(filePath, content) functions.

export default class MetaEditor {
  constructor({ api, readOnly = false }) {
    if (!api || typeof api.load !== 'function' || typeof api.save !== 'function') {
      throw new Error('MetaEditor requires { api: { load(filePath, cb), save(filePath, content) } }');
    }
    this.api = api;
    this.readOnly = !!readOnly;
    this.currentFilePath = null;
    this._listenerCleanup = null;

    this.modal = document.createElement('div');
    this.modal.className = 'modal';
    this.modal.innerHTML = `
      <div class="modal-background modal-close-action"></div>
      <div
        class="modal-card m-1"
        style="width: 90%; max-width: 1200px; height: calc(100vh - 8px); max-height: calc(100vh - 8px);"
      >
        <header class="modal-card-head py-2 px-3">
          <p class="modal-card-title is-size-4" id="meta_editor_title">Edit metadata</p>
          <button class="delete modal-close-action" aria-label="close"></button>
        </header>
        <section class="modal-card-body is-flex is-flex-direction-column is-flex-grow-1 p-2">
          <textarea
            id="meta_editor_textarea"
            class="textarea is-flex-grow-1 is-family-monospace is-size-5"
            style=" min-height: 0;
                    resize: none;
                    max-height: -webkit-fill-available;"
            placeholder="Loading..."
          ></textarea>
        </section>
        <footer class="modal-card-foot py-2 px-3 is-justify-content-flex-end">
          <div class="buttons">
            <button class="button is-primary modal-save-action">Save</button>
            <button class="button modal-close-action">Close</button>
          </div>
        </footer>
      </div>
    `;
    document.body.appendChild(this.modal);

    // Cache elements
    this.$title = this.modal.querySelector('#meta_editor_title');
    this.$textarea = this.modal.querySelector('#meta_editor_textarea');
    this.$saveBtn = this.modal.querySelector('.modal-save-action');

    // Close actions
    this.modal.querySelectorAll('.modal-close-action').forEach(el => {
      el.addEventListener('click', () => this.close());
    });

    // Save
    this.$saveBtn.addEventListener('click', async () => {
      if (!this.currentFilePath || this.readOnly) return;
      const text = this.$textarea.value ?? '';
      try {
        await this.api.save(this.currentFilePath, text);
      } catch (e) {
        console.error('MetaEditor save failed:', e);
      }
      this.close();
    });

    // Apply initial mode
    this._applyReadOnly(this.readOnly);
  }

  _applyReadOnly(flag) {
    this.readOnly = !!flag;
    this.$textarea.readOnly = this.readOnly;
    this.$textarea.classList.toggle('is-static', this.readOnly);
    this.$textarea.classList.toggle('has-text-grey', this.readOnly);
    // Optional subtle style for read-only
    if (this.readOnly) {
      this.$textarea.style.backgroundColor = 'var(--bulma-scheme-main-bis, #f5f5f5)';
    } else {
      this.$textarea.style.backgroundColor = '';
    }
    if (this.$saveBtn) {
      this.$saveBtn.style.display = this.readOnly ? 'none' : '';
      this.$saveBtn.disabled = this.readOnly;
    }
    this.$title.textContent = this.readOnly ? 'View metadata' : 'Edit metadata';
  }

  async open({ filePath, displayName, readOnly } = {}) {
    this.currentFilePath = filePath;
    // Allow per-open override; fallback to constructor flag
    if (typeof readOnly === 'boolean') this._applyReadOnly(readOnly);
    this.$title.textContent = `${displayName || filePath}${this.readOnly ? ' (read-only)' : ''}`;

    // Reset UI
    this.$textarea.value = 'Loading...';

    // Load content
    let cancelListener = null;
    const onLoaded = (content) => {
      console.log('MetaEditor loaded content for', filePath);
      this.$textarea.value = content || '';
      this.$textarea.placeholder = 'No .meta content for this file available yet.';
    };
    try {
      cancelListener = this.api.load(this.currentFilePath, onLoaded);
    } catch (e) {
      console.error('MetaEditor load failed:', e);
      onLoaded('Error loading metadata');
    }
    this._listenerCleanup = typeof cancelListener === 'function' ? cancelListener : null;

    // Show modal
    this.modal.classList.add('is-active');
    document.documentElement.classList.add('is-clipped');
  }

  close() {
    this.modal.classList.remove('is-active');
    document.documentElement.classList.remove('is-clipped');
    if (this._listenerCleanup) {
      try { this._listenerCleanup(); } catch {}
      this._listenerCleanup = null;
    }
    this.currentFilePath = null;
  }
}