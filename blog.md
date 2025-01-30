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
    {% assign sorted_posts = site.blogs | sort: 'date' | reverse %}
    {% for post in sorted_posts %}
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
            <div class="meta-left">
              <span class="date">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="meta-icon">
                  <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                  <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
                </svg>
                {{ post.date | date: "%B %-d, %Y" }}
              </span>
            </div>
            <div class="meta-right">
              <span class="reading-time">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="meta-icon">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                {% capture words %}{{ post.content | number_of_words }}{% endcapture %}
                {% unless words contains "-" %}
                  {{ words | plus: 250 | divided_by: 250 | append: " minute read" }}
                {% endunless %}
              </span>
            </div>
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
            <div class="categories">
              {% if post.categories.size > 0 %}
                {% for category in post.categories %}
                  <span class="category">{{ category }}</span>
                {% endfor %}
              {% endif %}
            </div>
            
            <a href="{{ post.url | relative_url }}" class="read-more">
              Continue reading
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="arrow-icon">
                <line x1="5" y1="12" x2="19" y2="12"></line>
                <polyline points="12 5 19 12 12 19"></polyline>
              </svg>
            </a>
          </div>
        </div>
      </div>
    {% endfor %}
  </div>
</div>