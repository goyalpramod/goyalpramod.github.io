document.addEventListener('DOMContentLoaded', () => {
  const article = document.querySelector('article');
  const tocNav = document.getElementById('toc');
  const tocWrapper = document.querySelector('.toc-wrapper');
  const tocToggle = document.getElementById('toc-toggle');
  const tocClose = document.querySelector('.toc-close');
  
  if (!article || !tocNav) return;

  // Updated to include h4
  const headings = article.querySelectorAll('h2, h3, h4');
  
  if (headings.length === 0) {
    tocToggle.style.display = 'none';
    return;
  }
  
  // Rest of the code remains the same
  tocToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    tocWrapper.classList.toggle('active');
  });

  tocClose.addEventListener('click', () => {
    tocWrapper.classList.remove('active');
  });

  document.addEventListener('click', (e) => {
    if (!tocWrapper.contains(e.target) && !tocToggle.contains(e.target)) {
      tocWrapper.classList.remove('active');
    }
  });
  
  const ul = document.createElement('ul');
  
  headings.forEach((heading, index) => {
    if (!heading.id) {
      heading.id = `heading-${index}`;
    }
    
    const li = document.createElement('li');
    const a = document.createElement('a');
    
    a.href = `#${heading.id}`;
    a.textContent = heading.textContent;
    a.classList.add(`toc-${heading.tagName.toLowerCase()}`);
    
    a.addEventListener('click', (e) => {
      e.preventDefault();
      heading.scrollIntoView({ behavior: 'smooth' });
      window.history.pushState(null, null, `#${heading.id}`);
    });
    
    li.appendChild(a);
    ul.appendChild(li);
  });
  
  tocNav.appendChild(ul);
  
  const observerCallback = (entries) => {
    entries.forEach(entry => {
      const id = entry.target.getAttribute('id');
      const tocItem = tocNav.querySelector(`a[href="#${id}"]`);
      
      if (entry.isIntersecting) {
        tocNav.querySelectorAll('a').forEach(a => a.classList.remove('active'));
        if (tocItem) tocItem.classList.add('active');
      }
    });
  };
  
  const observer = new IntersectionObserver(observerCallback, {
    rootMargin: '-70px 0px -70% 0px'
  });
  
  headings.forEach(heading => observer.observe(heading));
});