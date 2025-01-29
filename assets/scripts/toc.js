document.addEventListener('DOMContentLoaded', () => {
  const article = document.querySelector('article');
  const tocNav = document.getElementById('toc');
  const tocWrapper = document.querySelector('.toc-wrapper');
  const tocToggle = document.getElementById('toc-toggle');
  const tocClose = document.querySelector('.toc-close');
  
  if (!article || !tocNav) return;

  // Get all h2 and h3 headings from the article
  const headings = article.querySelectorAll('h2, h3');
  
  if (headings.length === 0) {
    tocToggle.style.display = 'none';
    return;
  }
  
  // Toggle ToC visibility with the main button
  tocToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    tocWrapper.classList.toggle('active');
  });

  tocClose.addEventListener('click', () => {
    tocWrapper.classList.remove('active');
  });

  // Close ToC when clicking outside
  document.addEventListener('click', (e) => {
    if (!tocWrapper.contains(e.target) && !tocToggle.contains(e.target)) {
      tocWrapper.classList.remove('active');
    }
  });
  
  // Create ToC structure
  const ul = document.createElement('ul');
  
  headings.forEach((heading, index) => {
    // Add id to heading if it doesn't have one
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
  
  // Highlight current section while scrolling
  const observerCallback = (entries) => {
    entries.forEach(entry => {
      const id = entry.target.getAttribute('id');
      const tocItem = tocNav.querySelector(`a[href="#${id}"]`);
      
      if (entry.isIntersecting) {
        // Remove active class from all links
        tocNav.querySelectorAll('a').forEach(a => a.classList.remove('active'));
        // Add active class to current link
        if (tocItem) tocItem.classList.add('active');
      }
    });
  };
  
  const observer = new IntersectionObserver(observerCallback, {
    rootMargin: '-70px 0px -70% 0px'
  });
  
  headings.forEach(heading => observer.observe(heading));
}); 