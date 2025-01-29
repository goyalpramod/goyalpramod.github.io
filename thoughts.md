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

<style>
.thoughts-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}

.thoughts-header {
  text-align: center;
  padding: 2rem 0;
  margin-bottom: 2rem;
}

.page-title {
  color: #5C4B37;
  font-size: 3rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  letter-spacing: -0.02em;
}

.page-description {
  color: #8B7355;
  font-size: 1.2rem;
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
}

.thoughts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
}

.thought-card {
  background: #F7EFE2;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(139, 115, 85, 0.05);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid rgba(222, 184, 135, 0.4);
  overflow: hidden;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.thought-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(139, 115, 85, 0.12);
  border-color: rgba(211, 84, 0, 0.3);
}

.thought-image-link {
  display: block;
  text-decoration: none;
}

.thought-image-container {
  width: 100%;
  padding-top: 56.25%; /* 16:9 aspect ratio */
  position: relative;
  overflow: hidden;
}

.thought-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.thought-card:hover .thought-image {
  transform: scale(1.05);
}

.thought-content {
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  flex-grow: 1;
}

.thought-meta {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  color: #8B7355;
  font-size: 0.95rem;
  margin-bottom: 1rem;
}

.update-date, .reading-time {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.meta-icon {
  color: #D35400;
  flex-shrink: 0;
}

.thought-title {
  margin: 0 0 1rem 0;
}

.thought-title a {
  color: #D35400;
  font-size: 1.5rem;
  text-decoration: none;
  font-weight: 600;
  line-height: 1.3;
  transition: color 0.3s ease;
}

.thought-title a:hover {
  color: #E67E22;
}

.thought-excerpt {
  color: #5C4B37;
  line-height: 1.6;
  margin-bottom: 1.5rem;
  opacity: 0.85;
}

.thought-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: auto;
}

.read-more {
  color: #D35400;
  text-decoration: none;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.2s ease;
}

.read-more:hover {
  color: #E67E22;
  transform: translateX(5px);
}

.arrow-icon {
  transition: transform 0.2s ease;
}

.read-more:hover .arrow-icon {
  transform: translateX(3px);
}

.categories {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.category {
  background: rgba(211, 84, 0, 0.1);
  color: #D35400;
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.85rem;
  white-space: nowrap;
}

.no-thoughts {
  grid-column: 1 / -1;
  text-align: center;
  padding: 3rem;
  background: #F7EFE2;
  border-radius: 12px;
  color: #8B7355;
  font-size: 1.1rem;
}

@media (max-width: 768px) {
  .page-title {
    font-size: 2.5rem;
  }
  
  .thoughts-grid {
    grid-template-columns: 1fr;
  }
  
  .thought-meta {
    flex-wrap: wrap;
  }
  
  .thought-footer {
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
  }
}
</style>