document.addEventListener('DOMContentLoaded', () => {
    // Create dark mode toggle button
    const darkModeButton = document.createElement('button');
    darkModeButton.id = 'dark-mode-toggle';
    darkModeButton.setAttribute('aria-label', 'Toggle dark mode');
    darkModeButton.innerHTML = '<i class="fas fa-moon"></i>';
    document.body.appendChild(darkModeButton);
  
    // Dark mode functionality
    const savedTheme = localStorage.getItem('theme') || 
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateDarkModeIcon(savedTheme === 'dark');
  
    darkModeButton.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
      updateDarkModeIcon(newTheme === 'dark');
      
      // Update utterances theme if it exists
      const utterances = document.querySelector('.utterances-frame');
      if (utterances) {
        const utterancesTheme = newTheme === 'dark' ? 'github-dark' : 'github-light';
        const message = {
          type: 'set-theme',
          theme: utterancesTheme
        };
        utterances.contentWindow.postMessage(message, 'https://utteranc.es');
      }
    });
  
    function updateDarkModeIcon(isDark) {
      darkModeButton.innerHTML = isDark ? 
        '<i class="fas fa-sun"></i>' : 
        '<i class="fas fa-moon"></i>';
    }
  
    // Handle system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      if (!localStorage.getItem('theme')) {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
        updateDarkModeIcon(e.matches);
      }
    });
  
    // Back to top functionality
    const backToTopButton = document.getElementById('back-to-top');
    if (backToTopButton) {
      // Show/hide based on scroll position
      const toggleBackToTop = () => {
        if (window.scrollY > 400) {
          backToTopButton.classList.add('visible');
        } else {
          backToTopButton.classList.remove('visible');
        }
      };
  
      // Smooth scroll to top
      const scrollToTop = () => {
        const currentPosition = document.documentElement.scrollTop || document.body.scrollTop;
        
        if (currentPosition > 0) {
          window.requestAnimationFrame(scrollToTop);
          window.scrollTo(0, currentPosition - currentPosition / 8);
        }
      };
  
      window.addEventListener('scroll', toggleBackToTop);
      backToTopButton.addEventListener('click', (e) => {
        e.preventDefault();
        scrollToTop();
      });
    }
  });