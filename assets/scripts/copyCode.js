document.addEventListener('DOMContentLoaded', () => {
    // Remove any existing copy buttons first
    document.querySelectorAll('.copy-code-button').forEach(button => button.remove());
    
    const codeBlocks = document.querySelectorAll('.highlighter-rouge');
    
    codeBlocks.forEach((codeBlock) => {
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-code-button';
      copyButton.setAttribute('aria-label', 'Copy code to clipboard');
      copyButton.innerHTML = '<i class="far fa-clone"></i>';
      
      const code = codeBlock.querySelector('pre').innerText;
      
      copyButton.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(code);
          
          copyButton.classList.add('copied');
          const originalIcon = copyButton.innerHTML;
          copyButton.innerHTML = '<i class="fas fa-check"></i>';
          
          setTimeout(() => {
            copyButton.classList.remove('copied');
            copyButton.innerHTML = originalIcon;
          }, 2000);
        } catch (err) {
          console.error('Failed to copy:', err);
        }
      });
      
      codeBlock.insertBefore(copyButton, codeBlock.firstChild);
    });
  });