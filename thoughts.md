---
layout: page
title: Thoughts
permalink: /thoughts/
---

Here's where I share my random thoughts, believes, philosophy etc.

<ul class="list-unstyled">
  {% for thought in site.thoughts %}
    <li class="mb-4">
      <h3>
        <a href="{{ thought.url | relative_url }}">{{ thought.title | escape }}</a>
      </h3>
      <p class="text-muted">
        Last updated: {{ thought.last_modified_at | date: "%B %-d, %Y" }}
      </p>
      {% if site.show_excerpts %}
        {{ thought.excerpt }}
      {% endif %}
    </li>
  {% else %}
    <li>No thoughts found</li>
  {% endfor %}
</ul> 