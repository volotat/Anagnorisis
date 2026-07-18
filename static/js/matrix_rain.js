/* =====================================================================
   matrix_rain.js — background effects for the "Matrix" theme.

   Self-contained companion to /static/css/theme_matrix.css:
   • Falling katakana rain on a full-screen canvas behind the content
     (the canvas is created only while data-theme="matrix" is active,
     and removed the moment another theme is selected).
   • Rare "glitch" bursts on random visible images (adds the CSS class
     .matrix-glitching for ~0.5 s every 8–20 s).

   Zero effect on other themes: everything is gated on the data-theme
   attribute of <html>, observed via MutationObserver. Respects
   prefers-reduced-motion (static backdrop, no rain, no glitches).

   PERFORMANCE
   ───────────
   The renderer is built to stay cheap:
   • Half-resolution backing store (RENDER_SCALE) upscaled by CSS —
     4× fewer pixels for the per-frame trail fade and compositing;
     the slight softness reads as phosphor glow.
   • Glyphs are rasterized ONCE into an offscreen atlas (per color
     variant); frames blit cells with drawImage instead of calling
     fillText, avoiding font rasterization in the hot loop.
   • Opaque 2D context ({alpha:false}) — no alpha blending of the
     canvas against the page backdrop.
   • ~12 fps via rAF gating (rAF also auto-pauses in hidden tabs).
   ===================================================================== */

(function () {
  'use strict';

  const THEME = 'matrix';
  const CANVAS_ID = 'matrix-rain-canvas';

  // Katakana (as in the film's mirrored rain) + digits + a few symbols
  const GLYPHS =
    'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホ' +
    'マミムメモヤユヨラリルレロワヲンヴガギグゲゴザジズゼゾダヂヅデド' +
    '0123456789Z:・"=*+-<>¦｜';

  const FONT_SIZE = 26;        // big glyphs (CSS px), like the original rain
  const RENDER_SCALE = 0.5;    // internal resolution factor (0.5 = 4× fewer px)
  const FRAME_MS = 100;        // ~10 fps — calm fall, easy on eyes/CPU
  const TRAIL_ALPHA = 0.09;    // per-frame fade; lower = longer trails
  // 8-bit blending stalls once a pixel is within ~5 counts of the base
  // color (round(v*0.91 + base*0.09) == v), so glyphs would leave faint
  // permanent ghosts in a grid pattern. A periodic stronger fade pass
  // pushes the residue down to ~1 count, which is invisible.
  const DEEP_CLEAN_EVERY = 20; // frames between deep-clean passes
  const DEEP_CLEAN_ALPHA = 0.28;
  const BASE_COLOR = '#020804';
  // Atlas rows: one body color per column speed (1..3), last row = bright head
  const ROW_STYLES = [
    'rgba(0, 255, 65, 0.55)',
    'rgba(0, 255, 65, 0.40)',
    'rgba(0, 255, 65, 0.25)',
    'rgba(190, 255, 210, 0.85)'
  ];
  const HEAD_ROW = 3;
  const GLITCH_MIN_MS = 8000;  // glitches are rare on purpose
  const GLITCH_MAX_MS = 20000;

  const CELL = Math.round(FONT_SIZE * RENDER_SCALE);      // glyph cell, device px
  const CELL_H = Math.ceil(CELL * 1.15);                  // room for ascenders

  let canvas = null;
  let ctx = null;
  let atlas = null;
  let rafId = null;
  let lastFrame = 0;
  let columns = [];
  let glitchTimer = null;
  let running = false;
  let frameCount = 0;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  function isThemeActive() {
    return document.documentElement.getAttribute('data-theme') === THEME;
  }

  // ── Rain ────────────────────────────────────────────────────────────

  function buildAtlas() {
    atlas = document.createElement('canvas');
    atlas.width = CELL * GLYPHS.length;
    atlas.height = CELL_H * ROW_STYLES.length;
    const a = atlas.getContext('2d');
    a.font = CELL + 'px monospace';
    for (let row = 0; row < ROW_STYLES.length; row++) {
      a.fillStyle = ROW_STYLES[row];
      for (let i = 0; i < GLYPHS.length; i++) {
        a.fillText(GLYPHS.charAt(i), i * CELL, row * CELL_H + CELL);
      }
    }
  }

  function makeColumn(x) {
    return {
      x: x,
      y: Math.floor(Math.random() * -50),        // stagger entry from above
      speed: 1 + Math.floor(Math.random() * 3),  // advance every 1..3 frames
      tick: 0
    };
  }

  function initColumns() {
    const count = Math.ceil(canvas.width / CELL);
    columns = [];
    for (let i = 0; i < count; i++) columns.push(makeColumn(i));
  }

  function resize() {
    if (!canvas) return;
    canvas.width = Math.ceil(window.innerWidth * RENDER_SCALE);
    canvas.height = Math.ceil(window.innerHeight * RENDER_SCALE);
    ctx.fillStyle = BASE_COLOR;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    initColumns();
    // Pre-warm: simulate a few seconds of fall so the screen shows a
    // full rain field immediately instead of an empty black backdrop
    for (let i = 0; i < 150; i++) step();
  }

  function step() {
    // Fade previous frame toward the base color → trails
    frameCount++;
    const fade = (frameCount % DEEP_CLEAN_EVERY === 0) ? DEEP_CLEAN_ALPHA : TRAIL_ALPHA;
    ctx.fillStyle = 'rgba(2, 8, 4, ' + fade + ')';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < columns.length; i++) {
      const col = columns[i];
      col.tick++;
      if (col.tick % col.speed !== 0) continue;

      const glyph = Math.floor(Math.random() * GLYPHS.length);
      // Occasional bright "head" glyph sells the leading-edge shimmer;
      // slower columns use dimmer atlas rows — cheap depth/parallax
      const row = (Math.random() < 0.10) ? HEAD_ROW : (col.speed - 1);
      const py = col.y * CELL;

      ctx.drawImage(atlas,
        glyph * CELL, row * CELL_H, CELL, CELL_H,
        col.x * CELL, py - CELL, CELL, CELL_H);

      col.y++;
      // Reset with randomness once past the bottom so columns desync
      if (py > canvas.height && Math.random() > 0.975) {
        col.y = 0;
        col.speed = 1 + Math.floor(Math.random() * 3);
      }
    }
  }

  function drawFrame(now) {
    rafId = requestAnimationFrame(drawFrame);
    if (now - lastFrame < FRAME_MS) return;
    lastFrame = now;
    step();
  }

  // ── Image glitches ──────────────────────────────────────────────────

  function glitchRandomImage() {
    if (!isThemeActive() || document.hidden || reducedMotion.matches) return;

    const viewH = window.innerHeight;
    const viewW = window.innerWidth;
    const candidates = Array.prototype.filter.call(document.images, function (img) {
      if (img.classList.contains('matrix-glitching')) return false;
      const r = img.getBoundingClientRect();
      return r.width > 40 && r.height > 40 &&
             r.bottom > 0 && r.top < viewH &&
             r.right > 0 && r.left < viewW;
    });
    if (!candidates.length) return;

    const img = candidates[Math.floor(Math.random() * candidates.length)];
    img.classList.add('matrix-glitching');
    setTimeout(function () { img.classList.remove('matrix-glitching'); }, 550);
  }

  function scheduleGlitch() {
    const delay = GLITCH_MIN_MS + Math.random() * (GLITCH_MAX_MS - GLITCH_MIN_MS);
    glitchTimer = setTimeout(function () {
      glitchRandomImage();
      scheduleGlitch();
    }, delay);
  }

  // ── Lifecycle ───────────────────────────────────────────────────────

  function start() {
    if (running) return;
    running = true;

    canvas = document.createElement('canvas');
    canvas.id = CANVAS_ID;
    canvas.setAttribute('aria-hidden', 'true');
    document.body.appendChild(canvas);
    // Opaque context: skips alpha compositing against the page backdrop
    ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
    if (!atlas) buildAtlas();
    resize();
    window.addEventListener('resize', resize);

    if (!reducedMotion.matches) {
      lastFrame = 0;
      rafId = requestAnimationFrame(drawFrame); // rAF auto-pauses on hidden tabs
      scheduleGlitch();
    }
  }

  function stop() {
    if (!running) return;
    running = false;

    window.removeEventListener('resize', resize);
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
    if (glitchTimer !== null) { clearTimeout(glitchTimer); glitchTimer = null; }
    if (canvas) { canvas.remove(); canvas = null; ctx = null; }

    Array.prototype.forEach.call(
      document.querySelectorAll('img.matrix-glitching'),
      function (img) { img.classList.remove('matrix-glitching'); }
    );
  }

  function sync() {
    if (isThemeActive()) start();
    else stop();
  }

  // React to theme switches (theme-switcher sets data-theme on <html>)
  new MutationObserver(sync).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme']
  });

  // Restart/stop effects if the user toggles reduced-motion while active
  const onMotionChange = function () {
    if (!isThemeActive()) return;
    stop();
    start();
  };
  if (typeof reducedMotion.addEventListener === 'function') {
    reducedMotion.addEventListener('change', onMotionChange);
  }

  // Initial state (the anti-flash snippet sets data-theme before this runs)
  if (document.body) sync();
  else document.addEventListener('DOMContentLoaded', sync);
})();
