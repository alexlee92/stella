---
title: Screen recordings
has_children: true
nav_order: 75
has_toc: false
description: Screen recordings of stella building stella.
highlight_image: /assets/recordings.jpg
---

# Screen recordings

Below are a series of screen recordings of the stella developer using stella
to enhance stella.
They contain commentary that describes how stella is being used,
and might provide some inspiration for your own use of stella.

{% assign sorted_pages = site.pages | where: "parent", "Screen recordings" | sort: "nav_order" %}
{% for page in sorted_pages %}
- [{{ page.title }}]({{ page.url | relative_url }}) - {{ page.description }}
{% endfor %}

