// Masonry grid + lightbox for /photography/
(function () {
  const grid = document.getElementById('photo-grid');
  const lightbox = document.getElementById('photo-lightbox');
  if (!grid || !lightbox) return;

  const items = Array.from(grid.querySelectorAll('.photo-item'));

  // --- Masonry ------------------------------------------------------------
  // The grid uses tiny implicit rows; each item spans as many rows as its
  // aspect ratio needs at the current column width.
  function layout() {
    const styles = getComputedStyle(grid);
    const rowHeight = parseFloat(styles.gridAutoRows);
    const gap = parseFloat(styles.rowGap);
    items.forEach(item => {
      const img = item.querySelector('img');
      // Use the attributes, not the rendered size: lazy images may not be loaded yet
      const ratio = img.getAttribute('height') / img.getAttribute('width');
      const height = item.getBoundingClientRect().width * ratio;
      item.style.gridRowEnd = 'span ' + Math.ceil((height + gap) / (rowHeight + gap));
    });
    grid.classList.add('is-ready');
  }

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(layout, 100);
  });
  layout();

  // --- Lightbox -----------------------------------------------------------
  const lbImage = lightbox.querySelector('.lb-image');
  const lbTitle = lightbox.querySelector('.lb-title');
  const lbWhere = lightbox.querySelector('.lb-where');
  const lbNote = lightbox.querySelector('.lb-note');
  const lbGear = lightbox.querySelector('.lb-gear');
  const lbSettings = lightbox.querySelector('.lb-settings');
  let current = 0;
  let lastFocus = null;

  function show(index) {
    current = (index + items.length) % items.length;
    const d = items[current].dataset;

    // Show the already-loaded thumbnail instantly, swap in the full image when ready
    lbImage.src = items[current].querySelector('img').currentSrc;
    lbImage.alt = d.title;
    const full = new Image();
    full.onload = () => { if (items[current].dataset.full === d.full) lbImage.src = d.full; };
    full.src = d.full;

    lbTitle.textContent = d.title;
    lbWhere.textContent = [d.location, d.date].filter(Boolean).join(' · ');
    lbWhere.hidden = !lbWhere.textContent;
    lbNote.textContent = d.note || 'Notes coming soon.';
    lbNote.classList.toggle('is-placeholder', !d.note);
    lbGear.textContent = d.gear;
    lbSettings.textContent = d.settings;
  }

  function open(index) {
    lastFocus = document.activeElement;
    show(index);
    lightbox.hidden = false;
    document.body.style.overflow = 'hidden';
    requestAnimationFrame(() => lightbox.classList.add('is-open'));
    lightbox.querySelector('.lb-close').focus();
  }

  function close() {
    lightbox.classList.remove('is-open');
    document.body.style.overflow = '';
    setTimeout(() => { lightbox.hidden = true; }, 200);
    if (lastFocus) lastFocus.focus();
  }

  items.forEach((item, i) => {
    item.addEventListener('click', () => open(i));
    item.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(i); }
    });
  });

  lightbox.querySelector('.lb-close').addEventListener('click', close);
  lightbox.querySelector('.lb-prev').addEventListener('click', () => show(current - 1));
  lightbox.querySelector('.lb-next').addEventListener('click', () => show(current + 1));
  lightbox.addEventListener('click', e => {
    if (e.target === lightbox || e.target.classList.contains('lb-image-wrap')) close();
  });

  document.addEventListener('keydown', e => {
    if (lightbox.hidden) return;
    if (e.key === 'Escape') close();
    if (e.key === 'ArrowLeft') show(current - 1);
    if (e.key === 'ArrowRight') show(current + 1);
  });

  // Swipe on touch screens
  let touchX = null;
  lightbox.addEventListener('touchstart', e => { touchX = e.touches[0].clientX; }, { passive: true });
  lightbox.addEventListener('touchend', e => {
    if (touchX === null) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 50) show(current + (dx < 0 ? 1 : -1));
    touchX = null;
  });
})();
