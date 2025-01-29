document.addEventListener('DOMContentLoaded', () => {
  const backToTopButton = document.getElementById('back-to-top');
  
  // Show button after scrolling down 400px
  const toggleBackToTop = () => {
    if (window.scrollY > 400) {
      backToTopButton.classList.add('visible');
    } else {
      backToTopButton.classList.remove('visible');
    }
  };

  // Smooth scroll to top with easing
  const scrollToTop = () => {
    const currentPosition = document.documentElement.scrollTop || document.body.scrollTop;
    
    if (currentPosition > 0) {
      window.requestAnimationFrame(scrollToTop);
      window.scrollTo(0, currentPosition - currentPosition / 8);
    }
  };

  // Add event listeners
  window.addEventListener('scroll', toggleBackToTop);
  backToTopButton.addEventListener('click', (e) => {
    e.preventDefault();
    scrollToTop();
  });
}); 