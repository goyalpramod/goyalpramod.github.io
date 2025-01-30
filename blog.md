---
layout: default
title: Blog
permalink: /blog/
---

<div class="blog-container">
  <header class="blog-header">
    <h1 class="page-title">Blog</h1>
    <p class="page-description">Here's where I share my insights about Machine Learning, AI, and Technology.</p>
  </header>
  
  <div class="posts-grid">
    {% for post in site.blogs %}
      <div class="post-card">
        {% if post.image %}
        <a href="{{ post.url | relative_url }}" class="post-image-link">
          <div class="post-image-container">
            <img src="{{ post.image | relative_url }}" alt="{{ post.title }}" class="post-image">
          </div>
        </a>
        {% endif %}
        
        <div class="post-content">
          <div class="post-meta">
            <span class="date">{{ post.date | date: "%B %-d, %Y" }}</span>
            <span class="separator">•</span>
            <span class="reading-time">
              {% capture words %}{{ post.content | number_of_words }}{% endcapture %}
              {% unless words contains "-" %}
                {{ words | plus: 250 | divided_by: 250 | append: " minute read" }}
              {% endunless %}
            </span>
          </div>
          
          <h2 class="post-title">
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </h2>
          
          <div class="post-excerpt">
            {% if post.excerpt %}
              {{ post.excerpt | strip_html | truncate: 160 }}
            {% endif %}
          </div>
          
          <div class="post-footer">
            <a href="{{ post.url | relative_url }}" class="read-more">
              Continue reading
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="arrow-icon">
                <line x1="5" y1="12" x2="19" y2="12"></line>
                <polyline points="12 5 19 12 12 19"></polyline>
              </svg>
            </a>
            
            {% if post.categories.size > 0 %}
            <div class="categories">
              {% for category in post.categories %}
                <span class="category">{{ category }}</span>
              {% endfor %}
            </div>
            {% endif %}
          </div>
        </div>
      </div>
    {% endfor %}
  </div>
</div>
