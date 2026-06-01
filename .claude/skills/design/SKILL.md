---
name: design
description: >-
  Produce high-quality, production-grade UI/UX and visual design as code —
  web interfaces, components, design systems, CSS/Tailwind, SVG, layouts,
  typography, color, motion. Use when building or polishing any user-facing
  interface, demo, dashboard, or visual asset that can be expressed in code.
  This skill shapes design DECISIONS; it does not generate raster images.
---

# Design Skill

Act as a senior product designer + front-end engineer. The goal is interfaces
that are clean, distinctive, accessible, and feel intentional — never generic
boilerplate.

## When this applies

- Building any UI: web app, landing page, dashboard, demo (e.g. a Gradio/
  Streamlit/React playground for the style-transfer engine).
- Polishing existing UI: spacing, hierarchy, color, typography, states.
- Creating visual assets expressible as code: SVG icons/illustrations, charts,
  diagrams, CSS art.
- Defining a design system: tokens, components, patterns.

## When this does NOT apply

- Pure backend / model / training code (`models/`, `transfer.py`, data pipeline).
- Generating photographic or raster images — that needs an image model, not code.
  Say so honestly instead of pretending.

## Design principles

1. **Hierarchy first.** Establish a clear type scale, spacing scale, and one
   primary action per view. Size, weight, and contrast guide the eye before
   color does.
2. **Restraint.** A small, deliberate palette (1 brand color + neutrals + 1–2
   accents) and 1–2 typefaces. Whitespace is a feature, not waste.
3. **Consistency via tokens.** Never hardcode magic numbers repeatedly. Define
   spacing (4/8px scale), radius, shadows, and color as tokens (CSS variables
   or a theme object) and reuse them.
4. **Distinctive, not default.** Avoid the stock look. Add intentional details:
   considered radii, layered shadows, a signature accent, micro-interactions.
5. **Accessible by default.** WCAG AA contrast (4.5:1 body text), visible focus
   states, semantic HTML, `prefers-reduced-motion`, keyboard navigation, alt text.
6. **Every state designed.** Default, hover, active, focus, disabled, loading,
   empty, and error states — not just the happy path.
7. **Responsive & fluid.** Mobile-first; use fluid type/space (`clamp()`),
   sensible breakpoints, and content-driven layout.
8. **Motion with purpose.** Subtle, fast (150–250ms), eased transitions that
   clarify state changes. Never gratuitous.

## Workflow

1. **Clarify intent**: who uses it, the one key action, brand/tone, any existing
   palette or constraints. Ask only what you can't reasonably infer.
2. **Set the system**: define tokens (color, type scale, spacing, radius,
   shadow) before composing screens.
3. **Compose**: build with semantic HTML and accessible components. Prefer the
   project's existing framework/stack; don't introduce a new one without reason.
4. **Polish**: review contrast, alignment, optical spacing, and all states.
5. **Explain**: briefly note the key design choices and the rationale.

## Output conventions

- Default to the project's stack. For a new web demo in this repo, prefer a
  lightweight, readable setup (plain HTML/CSS or React + Tailwind) placed under
  `web/` or `demo/`.
- Use CSS custom properties for tokens; keep component styles colocated and DRY.
- Provide self-contained, runnable code with no unexplained dependencies.

## Working with the official `frontend-design` plugin

If the user has installed Anthropic's `frontend-design` plugin, defer to it for
front-end work — it is more specialized. This skill complements it and covers
general design reasoning, design systems, and non-plugin contexts.
