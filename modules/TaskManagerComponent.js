// Task Manager UI — badge in navbar + modal with task list
(function () {
  // ---- state ----
  let state = { active: null, queued: [], history: [], schedulers: [] };

  // ---- helpers ----
  function ago(ts) {
    if (!ts) return '';
    const s = Math.round((Date.now() / 1000) - ts);
    if (s < 60) return 'just now';
    if (s < 3600) return Math.floor(s / 60) + ' min ago';
    if (s < 86400) return Math.floor(s / 3600) + ' h ago';
    return Math.floor(s / 86400) + ' d ago';
  }

  function pct(v) { return Math.round((v || 0) * 100); }

  function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s || '';
    return d.innerHTML;
  }

  function formatCountdown(secondsLeft) {
    const s = Math.max(0, Math.round(secondsLeft));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    let parts = [];
    if (h > 0) parts.push(h + 'h');
    if (h > 0 || m > 0) parts.push(m + 'm');
    parts.push(sec + 's');
    return parts.join(' ');
  }

  // ---- rendering ----
  function taskCount() {
    let n = state.queued.length;
    if (state.active) n++;
    return n;
  }

  function updateBadge() {
    const badge = document.getElementById('task-manager-badge');
    if (!badge) return;
    const n = taskCount();
    badge.textContent = n;
    badge.style.display = n > 0 ? '' : 'none';
  }

  // Update only the countdown spans in-place (no full re-render needed).
  function updateCountdowns() {
    const modal = document.getElementById('tm-modal');
    if (!modal || !modal.classList.contains('is-active')) return;

    document.querySelectorAll('[data-next-run-at]').forEach(function (el) {
      const nextRunAt = parseFloat(el.getAttribute('data-next-run-at'));
      if (isNaN(nextRunAt)) return;
      const remaining = nextRunAt - (Date.now() / 1000);
      el.textContent = remaining > 0 ? formatCountdown(remaining) : '0s';
    });
  }

  function renderSchedulers() {
    const schedulers = state.schedulers || [];
    if (!schedulers.length)
      return '<p class="has-text-grey is-size-7 mb-0">No schedulers registered.</p>';

    // Group by module prefix (part before first ':')
    const groups = {};
    schedulers.forEach(function (s) {
      const sep = s.name.indexOf(':');
      const group = sep > 0 ? s.name.slice(0, sep).trim() : 'Other';
      if (!groups[group]) groups[group] = [];
      groups[group].push(s);
    });

    return Object.keys(groups).map(function (groupName) {
      const rows = groups[groupName].map(function (s) {
        let countdownHtml;
        if (s.paused) {
          countdownHtml = '<span class="tag is-warning is-light is-small">Paused</span>';
        } else if (s.next_run_at === null || s.next_run_at === undefined) {
          countdownHtml = '<span class="tag is-info is-light is-small">Running\u2026</span>';
        } else {
          const remaining = s.next_run_at - (Date.now() / 1000);
          const display = remaining > 0 ? formatCountdown(remaining) : '0s';
          countdownHtml = `<span class="is-size-7 has-text-grey" data-next-run-at="${s.next_run_at}">${escHtml(display)}</span>`;
        }

        const taskLabel = s.name.indexOf(':') > 0 ? s.name.slice(s.name.indexOf(':') + 1).trim() : s.name;

        const pauseBtn = s.paused
          ? `<button class="button is-small is-warning is-outlined" onclick="window._tm.resumeScheduler('${s.id}')"><span class="icon is-small"><i class="fas fa-play"></i></span><span>Resume</span></button>`
          : `<button class="button is-small is-info is-outlined" onclick="window._tm.pauseScheduler('${s.id}')"><span class="icon is-small"><i class="fas fa-pause"></i></span><span>Pause</span></button>`;

        return `
        <div class="box mb-2 py-2 px-4">
          <div class="is-flex is-justify-content-space-between is-align-items-center">
            <div style="flex:1;min-width:0;margin-right:0.75rem;">
              <span class="is-size-7 has-text-weight-semibold" style="display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escHtml(s.name)}">${escHtml(taskLabel)}</span>
              <span class="is-size-7 has-text-grey">every ${s.interval_minutes}m &nbsp;&middot;&nbsp; next: ${countdownHtml}</span>
            </div>
            <div class="buttons are-small mb-0" style="flex-shrink:0;">
              ${pauseBtn}
            </div>
          </div>
        </div>`;
      }).join('');

      return `
        <p class="is-size-7 has-text-weight-semibold has-text-grey-dark mb-1 mt-2">${escHtml(groupName)}</p>
        ${rows}`;
    }).join('');
  }

  function renderActive() {
    const t = state.active;
    if (!t) return '<p class="has-text-grey is-size-7 mb-0">No task running.</p>';

    const isPaused = t.status === 'paused';
    const progressClass = isPaused ? 'is-warning' : 'is-info';
    const pauseBtn = isPaused
      ? `<button class="button is-small is-warning is-outlined" onclick="window._tm.resume('${t.id}')"><span class="icon is-small"><i class="fas fa-play"></i></span><span>Resume</span></button>`
      : `<button class="button is-small is-info is-outlined"    onclick="window._tm.pause('${t.id}')"><span class="icon is-small"><i class="fas fa-pause"></i></span><span>Pause</span></button>`;

    return `
      <div class="box mb-3 py-3 px-4">
        <div class="is-flex is-justify-content-space-between is-align-items-center mb-2">
          <span class="has-text-weight-semibold">${escHtml(t.name)}</span>
          <span class="tag ${isPaused ? 'is-warning' : 'is-info'} is-light is-small">${isPaused ? 'Paused' : 'Running'}</span>
        </div>
        <progress class="progress ${progressClass} is-small mb-2" value="${pct(t.progress)}" max="100">${pct(t.progress)}%</progress>
        <div class="is-flex is-justify-content-space-between is-align-items-center">
          <span class="is-size-7 has-text-grey" style="width:calc(100% - 180px)">${escHtml(t.message)}  —  ${pct(t.progress)}%</span>
          <div class="buttons are-small mb-0">
            ${pauseBtn}
            <button class="button is-small is-danger is-outlined" onclick="window._tm.cancel('${t.id}')">
              <span class="icon is-small"><i class="fas fa-times"></i></span><span>Cancel</span>
            </button>
          </div>
        </div>
      </div>`;
  }

  function renderQueued() {
    if (!state.queued.length)
      return '<p class="has-text-grey is-size-7 mb-0">Queue is empty.</p>';

    return state.queued.map((t, idx) => `
      <div class="box mb-2 py-2 px-4">
        <div class="is-flex is-justify-content-space-between is-align-items-center">
          <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-right:0.75rem;">
            <span class="tag is-light is-small mr-2">#${idx + 1}</span>
            ${escHtml(t.name)}
          </span>
          <div class="buttons are-small mb-0" style="flex-shrink:0;">
            <button class="button is-small is-danger is-outlined" onclick="window._tm.remove('${t.id}')">
              <span class="icon is-small"><i class="fas fa-times"></i></span>
            </button>
          </div>
        </div>
      </div>`).join('');
  }

  function renderHistory() {
    if (!state.history.length)
      return '<p class="has-text-grey is-size-7 mb-0">No recent tasks.</p>';

    return state.history.slice(0, 15).map(t => {
      let icon, cls;
      if (t.status === 'completed')  { icon = 'fa-check-circle'; cls = 'has-text-success'; }
      else if (t.status === 'failed') { icon = 'fa-exclamation-circle'; cls = 'has-text-danger'; }
      else                             { icon = 'fa-ban'; cls = 'has-text-grey'; }

      let extra = '';
      if (t.status === 'failed' && t.error) {
        extra = `<pre class="is-size-7 mt-1 mb-0 p-2" style="max-height:6em;overflow:auto;white-space:pre-wrap;">${escHtml(t.error)}</pre>`;
      }

      return `
        <div class="is-flex is-align-items-center mb-2">
          <span class="icon is-small ${cls} mr-2"><i class="fas ${icon}"></i></span>
          <span class="is-size-7" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escHtml(t.name)}">${escHtml(t.name)}</span>
          <span class="is-size-7 has-text-grey ml-2" style="flex-shrink:0">${ago(t.finished_at)}</span>
        </div>${extra}`;
    }).join('');
  }

  function renderModal() {
    const body = document.getElementById('tm-modal-body');
    if (!body) return;

    body.innerHTML = `
      <h6 class="title is-6 mb-2">
        <span class="icon is-small mr-1"><i class="fas fa-play-circle"></i></span> Running
      </h6>
      ${renderActive()}

      <hr class="my-3">

      <h6 class="title is-6 mb-2">
        <span class="icon is-small mr-1"><i class="fas fa-clock"></i></span> Queued
        <span class="tag is-light is-small ml-1">${state.queued.length}</span>
      </h6>
      ${renderQueued()}

      <hr class="my-3">

      <h6 class="title is-6 mb-2">
        <span class="icon is-small mr-1"><i class="fas fa-history"></i></span> Recent
      </h6>
      ${renderHistory()}

      <hr class="my-3">

      <h6 class="title is-6 mb-2">
        <span class="icon is-small mr-1"><i class="fas fa-calendar-check"></i></span> Schedulers
        <span class="tag is-light is-small ml-1">${(state.schedulers || []).length}</span>
      </h6>
      ${renderSchedulers()}
    `;
  }

  // ---- socket wiring ----
  function applyState(s) {
    state = s;
    if (!state.schedulers) state.schedulers = [];
    updateBadge();
    // Only re-render if modal is open
    const modal = document.getElementById('tm-modal');
    if (modal && modal.classList.contains('is-active')) renderModal();
  }

  $(document).ready(function () {
    // Request initial state
    socket.emit('task_manager_get_state', {}, function (s) { applyState(s); });

    // Live updates
    socket.on('task_manager_update', function (s) { applyState(s); });

    // Tick countdowns every second (client-side, no server round-trip).
    setInterval(updateCountdowns, 1000);

    // Open / close modal
    document.getElementById('task-manager-button').addEventListener('click', function () {
      document.getElementById('tm-modal').classList.add('is-active');
      document.documentElement.classList.add('is-clipped');
      renderModal();
    });
    document.querySelectorAll('.tm-modal-close').forEach(function (el) {
      el.addEventListener('click', function () {
        document.getElementById('tm-modal').classList.remove('is-active');
        document.documentElement.classList.remove('is-clipped');
      });
    });

    // --- Test task buttons ---
    const addTestBtn = document.getElementById('tm-test-counter');
    if (addTestBtn) addTestBtn.addEventListener('click', function () {
      socket.emit('task_manager_submit_test', { type: 'counter', count: 20 });
    });
    const addFailBtn = document.getElementById('tm-test-fail');
    if (addFailBtn) addFailBtn.addEventListener('click', function () {
      socket.emit('task_manager_submit_test', { type: 'fail' });
    });
    const addInstantBtn = document.getElementById('tm-test-instant');
    if (addInstantBtn) addInstantBtn.addEventListener('click', function () {
      socket.emit('task_manager_submit_test', { type: 'instant' });
    });
  });

  // ---- public control API (called from onclick in rendered HTML) ----
  window._tm = {
    cancel:          function (id) { socket.emit('task_manager_cancel',           { task_id: id }); },
    pause:           function (id) { socket.emit('task_manager_pause',            { task_id: id }); },
    resume:          function (id) { socket.emit('task_manager_resume',           { task_id: id }); },
    remove:          function (id) { socket.emit('task_manager_remove',           { task_id: id }); },
    pauseScheduler:  function (id) { socket.emit('task_manager_pause_scheduler',  { scheduler_id: id }); },
    resumeScheduler: function (id) { socket.emit('task_manager_resume_scheduler', { scheduler_id: id }); },
  };
})();
