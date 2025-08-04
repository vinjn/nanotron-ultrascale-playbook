---
title: The Ultra-Scale Playbook
emoji: 🌌
colorFrom: yellow
colorTo: purple
sdk: static
pinned: true
license: apache-2.0
header: mini
app_file: dist/index.html
thumbnail: https://huggingface.co/spaces/nanotron/ultrascale-playbook/resolve/main/screenshot.png
short_description: The ultimate guide to training LLM on large GPU Clusters
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


Instruction to install and run locally

```bash
npm install
npm run build
npm run dev

// If you want to change something change it in src/....

// Once you are finished
npm run build
// And commit the dist folder
```

## Loading HTML fragments:
There are two way to load HTML fragments:
1. Compile them into html during build time
2. Fetch them and insert them during run-time

## When to use what
- Use compile time fragments only on parts which you want to ensure are seen by every user right after page load (e.g logo)
- Use run-time fragments for everything else so that the final HTML is of reasonable size (<1MB idealy)

## How to add a new fragment
- Add it to the `src/fragments` folder (e.g. `src/fragments/banner.html`)
- For run-time fragments, add {{{fragment-name}}} to appropriate place in `src/index.html` (e.g. {{{fragment-banner}}})
- For compile-time fragments, add <div id="fragment-name"></div> to `src/index.html` where you want to insert the fragment (e.g. <div id="fragment-banner"></div>)


## How to know which fragments are available
- Run `npm run dev` and look at the console for available fragments