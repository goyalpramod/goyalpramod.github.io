---
layout: default
title: Contact Me
permalink: /contact/
---

<div class="contact-container">
  <header class="contact-header">
    <h1 class="page-title">Contact Me</h1>
    <p class="page-description">Feel free to reach out! I'm always open to interesting conversations, collaborations, or just a friendly hello.</p>
  </header>
  
  <div class="contact-grid">
  <!-- Email Card -->
  <div class="contact-card">
    <div class="contact-icon">
      <i class="fa-solid fa-envelope"></i>
    </div>
    <h2 class="contact-method">Email</h2>
    <p class="contact-description">Drop me a mail, I will get back to you in this life.</p>
    <a href="mailto:{{ site.email }}" class="contact-link">
      {{ site.email }}
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="arrow-icon">
        <line x1="5" y1="12" x2="19" y2="12"></line>
        <polyline points="12 5 19 12 12 19"></polyline>
      </svg>
    </a>
  </div>

  <!-- X Card -->
  <div class="contact-card">
    <div class="contact-icon">
      <i class="fa-brands fa-x-twitter"></i>
    </div>
    <h2 class="contact-method">X</h2>
    <p class="contact-description">Drop a "Hi" in <s>twitter</s> X.</p>
    <a href="https://twitter.com/{{ site.twitter_username }}" class="contact-link" target="_blank" rel="noopener noreferrer">
      @{{ site.twitter_username }}
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="arrow-icon">
        <line x1="5" y1="12" x2="19" y2="12"></line>
        <polyline points="12 5 19 12 12 19"></polyline>
      </svg>
    </a>
  </div>

  <!-- LinkedIn Card -->
  <div class="contact-card">
    <div class="contact-icon">
      <i class="fa-brands fa-linkedin-in"></i>
    </div>
    <h2 class="contact-method">LinkedIn</h2>
    <p class="contact-description">Connect with me.</p>
    <a href="https://linkedin.com/in/{{site.linkedin_username}}" class="contact-link" target="_blank" rel="noopener noreferrer">
      {{site.linkedin_username}}
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="arrow-icon">
        <line x1="5" y1="12" x2="19" y2="12"></line>
        <polyline points="12 5 19 12 12 19"></polyline>
      </svg>
    </a>
  </div>

  <!-- YouTube Card -->
  <div class="contact-card">
    <div class="contact-icon">
      <i class="fa-brands fa-youtube"></i>
    </div>
    <h2 class="contact-method">YouTube</h2>
    <p class="contact-description">Consider watching my videos.</p>
    <a href="https://youtube.com/@{{ site.youtube_username }}" class="contact-link" target="_blank" rel="noopener noreferrer">
      @{{ site.youtube_username }}
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="arrow-icon">
        <line x1="5" y1="12" x2="19" y2="12"></line>
        <polyline points="12 5 19 12 12 19"></polyline>
      </svg>
    </a>
  </div>
</div>
</div>