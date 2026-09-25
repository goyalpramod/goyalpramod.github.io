document.addEventListener('DOMContentLoaded', () => {
  const backToTopButton = document.getElementById('back-to-top');
  
  if (!backToTopButton) return;
  
  const toggleBackToTop = () => {
    if (window.pageYOffset > 400) {
      backToTopButton.classList.add('visible');
    } else {
      backToTopButton.classList.remove('visible');
    }
  };

  const scrollToTop = () => {
    // Cancel any existing animation
    if (window.scrollAnimation) {
      cancelAnimationFrame(window.scrollAnimation);
    }

    const duration = 2000; // milliseconds
    const startPosition = window.pageYOffset;
    const startTime = performance.now();

    function easeOutQuint(t) {
      // Quintic easing out - smoother deceleration
      return 1 - Math.pow(1 - t, 5);
    }

    function animateScroll(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Using a single, smoother easing function
      const easeValue = easeOutQuint(progress);
      
      // Calculate new position with enhanced precision
      const position = Math.max(0, Math.floor(startPosition * (1 - easeValue)));
      
      // Only scroll if we're not at the top or animation isn't complete
      if (position > 0 || progress < 1) {
        window.scrollTo({
          top: position,
          behavior: 'auto' // Use auto to prevent conflicting smooth scrolling
        });
        window.scrollAnimation = requestAnimationFrame(animateScroll);
      } else {
        window.scrollTo({
          top: 0,
          behavior: 'auto'
        });
      }
    }

    // Start the animation
    window.scrollAnimation = requestAnimationFrame(animateScroll);
  };

  // Optimize scroll event listener with debouncing
  let scrollTimeout;
  window.addEventListener('scroll', () => {
    if (scrollTimeout) {
      window.cancelAnimationFrame(scrollTimeout);
    }
    
    scrollTimeout = window.requestAnimationFrame(() => {
      toggleBackToTop();
    });
  }, { passive: true });

  backToTopButton.addEventListener('click', (e) => {
    e.preventDefault();
    scrollToTop();
  });

  // Initial check
  toggleBackToTop();
});