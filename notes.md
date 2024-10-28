---
layout: page
title: Notes
permalink: /notes/
---

This is where I share my thoughts on various subjects.

<ul class="list-unstyled">
  {% for note in site.notes %}
    <li class="mb-4">
      <h3>
        <a href="{{ note.url | relative_url }}">{{ note.title | escape }}</a>
      </h3>
      <p class="text-muted">
        Last updated: {{ note.last_modified_at | date: "%B %-d, %Y" }}
      </p>
      {% if site.show_excerpts %}
        {{ note.excerpt }}
      {% endif %}
    </li>
  {% endfor %}
</ul>
