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

<style>
.blog-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}

.blog-header {
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

.posts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
}

.post-card {
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

.post-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(139, 115, 85, 0.12);
  border-color: rgba(211, 84, 0, 0.3);
}

.post-image-link {
  display: block;
  text-decoration: none;
}

.post-image-container {
  width: 100%;
  padding-top: 56.25%; /* 16:9 aspect ratio */
  position: relative;
  overflow: hidden;
}

.post-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.post-card:hover .post-image {
  transform: scale(1.05);
}

.post-content {
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  flex-grow: 1;
}

.post-meta {
  display: flex;
  align-items: center;
  color: #8B7355;
  font-size: 0.95rem;
  margin-bottom: 0.8rem;
}

.separator {
  margin: 0 0.5rem;
}

.post-title {
  margin: 0 0 1rem 0;
}

.post-title a {
  color: #D35400;
  font-size: 1.5rem;
  text-decoration: none;
  font-weight: 600;
  line-height: 1.3;
  transition: color 0.3s ease;
}

.post-title a:hover {
  color: #E67E22;
}

.post-excerpt {
  color: #5C4B37;
  line-height: 1.6;
  margin-bottom: 1.5rem;
  opacity: 0.85;
}

.post-footer {
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

@media (max-width: 768px) {
  .page-title {
    font-size: 2.5rem;
  }
  
  .posts-grid {
    grid-template-columns: 1fr;
  }
  
  .post-meta {
    flex-wrap: wrap;
  }
  
  .separator {
    display: none;
  }
  
  .reading-time {
    width: 100%;
    margin-top: 0.3rem;
  }
  
  .post-footer {
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
  }
}
</style>