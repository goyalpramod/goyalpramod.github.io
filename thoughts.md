---
layout: default
title: Thoughts
permalink: /thoughts/
---

<div class="thoughts-container">
  <header class="thoughts-header">
    <h1 class="page-title">Thoughts</h1>
    <p class="page-description">Here's where I share my random thoughts, beliefs, philosophy etc.</p>
  </header>
  
  <div class="thoughts-grid">
    {% for thought in site.thoughts %}
      <div class="thought-card">
        {% if thought.image %}
        <a href="{{ thought.url | relative_url }}" class="thought-image-link">
          <div class="thought-image-container">
            <img src="{{ thought.image | relative_url }}" alt="{{ thought.title }}" class="thought-image">
          </div>
        </a>
        {% endif %}
        
        <div class="thought-content">
          <div class="thought-meta">
            <span class="update-date">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="meta-icon">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
              </svg>
              Last updated: {{ thought.last_modified_at | date: "%B %-d, %Y" }}
            </span>
            <span class="reading-time">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="meta-icon">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
              </svg>
              {% capture words %}{{ thought.content | number_of_words }}{% endcapture %}
              {% unless words contains "-" %}
                {{ words | plus: 250 | divided_by: 250 | append: " minute read" }}
              {% endunless %}
            </span>
          </div>
          
          <h2 class="thought-title">
            <a href="{{ thought.url | relative_url }}">{{ thought.title }}</a>
          </h2>
          
          <div class="thought-excerpt">
            {% if thought.excerpt %}
              {{ thought.excerpt | strip_html | truncate: 160 }}
            {% endif %}
          </div>
          
          <div class="thought-footer">
            <a href="{{ thought.url | relative_url }}" class="read-more">
              Read more
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="arrow-icon">
                <line x1="5" y1="12" x2="19" y2="12"></line>
                <polyline points="12 5 19 12 12 19"></polyline>
              </svg>
            </a>
            
            {% if thought.categories.size > 0 %}
            <div class="categories">
              {% for category in thought.categories %}
                <span class="category">{{ category }}</span>
              {% endfor %}
            </div>
            {% endif %}
          </div>
        </div>
      </div>
    {% else %}
      <div class="no-thoughts">
        <p>No thoughts found</p>
      </div>
    {% endfor %}
  </div>
</div>
