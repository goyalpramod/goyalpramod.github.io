---
layout: page
title: Blog
permalink: /blog/
---

Here you'll find blogs on multiple ML topics.

<ul class="list-unstyled">
  {% for post in site.blogs %}
    <li class="mb-4">
      <h3>
        <a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
      </h3>
      <p class="text-muted">
        {{ post.date | date: "%B %-d, %Y" }} • 
        {% capture words %}{{ post.content | number_of_words }}{% endcapture %}
        {% unless words contains "-" %}
          {{ words | plus: 250 | divided_by: 250 | append: " minute read" }}
        {% endunless %}
      </p>
      {% if site.show_excerpts %}
        {{ post.excerpt }}
      {% endif %}
    </li>
  {% endfor %}
</ul>
