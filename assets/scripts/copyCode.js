// Create a self-executing function to avoid global scope pollution
(function() {
  // Keep track of initialization
  let initialized = false;
  
  function initializeCopyButtons() {
    // If already initialized, return immediately
    if (initialized) return;
    
    // First, remove any existing copy buttons
    document.querySelectorAll('.copy-code-button').forEach(button => button.remove());
    
    const codeBlocks = document.querySelectorAll('.highlighter-rouge');
    
    codeBlocks.forEach((codeBlock) => {
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-code-button';
      copyButton.setAttribute('aria-label', 'Copy code to clipboard');
      copyButton.setAttribute('type', 'button'); // Explicitly set button type
      
      // Create icon element
      const icon = document.createElement('i');
      icon.className = 'far fa-clone';
      copyButton.appendChild(icon);
      
      const code = codeBlock.querySelector('pre').innerText;
      
      copyButton.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(code);
          
          copyButton.classList.add('copied');
          icon.className = 'fas fa-check';
          
          setTimeout(() => {
            copyButton.classList.remove('copied');
            icon.className = 'far fa-clone';
          }, 2000);
        } catch (err) {
          console.error('Failed to copy:', err);
        }
      });
      
      // Only add if there isn't already a copy button
      if (!codeBlock.querySelector('.copy-code-button')) {
        codeBlock.insertBefore(copyButton, codeBlock.firstChild);
      }
    });
    
    initialized = true;
  }

  // Wait for DOM to be fully loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCopyButtons);
  } else {
    initializeCopyButtons();
  }
})();