# Intel Agent Skills — catalog page (site source)

This branch holds **only** the Astro project that builds the public catalog UI.
The skill registry (`skills.yaml`, `skills/*`) lives on **`main`**.

GitHub Actions on **`main`** checks out both branches, sets `CATALOG_ROOT` to the
`main` tree, and deploys to GitHub Pages (`SITE_BASE=/skills/`).

## Local development

Use a second worktree (or clone) for the catalog on `main`, then point the build at it:

```sh
git worktree add ../skills-catalog main
npm install
CATALOG_ROOT="$(pwd)/../skills-catalog" npm run dev
```

Build:

```sh
CATALOG_ROOT="$(pwd)/../skills-catalog" npm run build
```

Optional status badges: `npm run dev:with-status` / `npm run build:with-status`.

## Motion

Four effects, all decorative and all reversible. Deliberately no cursor-tracking effects:
a spotlight and a 3D tilt were both tried and removed — over the blurred backdrop each
halved the frame rate during pointer movement, and the tilt in particular is invisible
without a `perspective` on its wrapper.

- **Aurora backdrop** (`src/components/Aurora.astro`) — three overlapping colour fields
  echoing the glow Intel puts behind its hero silicon. Static, and
  `aria-hidden` with pointer events off.
- **Card hover** — a 3px lift with an accent border and shadow. Pure CSS; there is no
  pointer-tracking JavaScript on the grid.
- **Staggered entrance** for the header, cards and footer. The delay is capped at twelve
  steps — a 33rd card at 45 ms each would keep the reader waiting 1.5 s.
- **Dialog entrance**, fading and lifting in with its backdrop blur.
- **Copy confirmation**, a short pop on the button.

### Why the backdrop does not move

Drifting it measured 32 fps against 60 with the motion removed: a large `filter: blur()`
is re-rasterised on every frame it moves. It also compounded with the dialog's
`backdrop-filter`, which then had to re-blur a backdrop that never stopped changing —
16 fps with a dialog open, though `backdrop-filter` on its own costs nothing.

Held still it is free, but **only because of `will-change: transform` on the fields**.
That is not a hint about future animation; nothing there animates. It promotes each field
to its own layer so the blur is rasterised once and cached. Removing it measured 31 fps —
worse than animating it with the promotion in place. If you touch that rule, re-measure.

(Figures are from headless Chromium, which rasterises on the CPU via SwiftShader, so blur
is punished harder there than on real hardware. The ranking holds; the absolute numbers
do not.)

Two more details worth keeping if you edit this:

- The grid gets a `grid--settled` class once the entrance animations report finished.
  Filtering toggles `hidden`, and leaving `display: none` *restarts* an element's
  animations — without settling, every filter change replays the whole staggered
  entrance, which drags badly while typing a search.
- The `prefers-reduced-motion` block zeroes `animation-delay` as well as duration. Delay
  alone would leave staggered cards blank for half a second before appearing.

## Relabelling skill status

`skills.yaml` tracks maturity as `draft | incubating | validated | promoted`. The page
never shows those words; it shows the labels in **`src/lib/status.ts`**, which is the only
place they are named:

| skills.yaml | shown as |
| --- | --- |
| `draft` | Experimental |
| `incubating` | Preview |
| `published` | Preview (default when status omitted) |
| `validated` | Pre-release |
| `promoted` | Release |

To relabel — say to alpha / beta / release — edit the `label` strings in that file and
nothing else. Cards, the detail dialog, the filter chips and the search index all read
from it, so the change lands everywhere at once.

`tone` is kept separate from `label` on purpose: it names the maturity step rather than
the wording, so relabelling never disturbs the colours. The tones are aliased to existing
badge palettes in `src/styles/global.css` (`--badge-experimental-*` and friends); repoint
one there to recolour a step. A status in `skills.yaml` with no entry in `status.ts`
renders verbatim and warns during the build rather than disappearing.

## Search and filtering

Both run client-side over the pre-rendered grid; no results are fetched.

- **Filter chips** are generated from the same badge dimensions, one group per field.
  Selections within a group are OR'd, groups are AND'd together, and search is AND'd on
  top. Each chip carries a count recomputed against the *other* active filters, so
  options within a group stay additive; a chip whose count reaches zero is disabled.
  The `Validated on` group only renders when some skill sets `intel-hw-validated-on`.
- **Search** matches every whitespace-separated term against the skill name, the summary,
  and the badge values. It deliberately does *not* index the rest of the description:
  descriptions end in trigger and scope guidance that names sibling skills ("use
  `vllm-xpu-run` instead"), which made a search for a runtime return every skill that
  merely cross-references it.

## Data source

The catalog is read at **build time** from the checkout named by **`CATALOG_ROOT`**
(see `astro.config.mjs`). In CI that is the `main` branch; locally, use a worktree as above.

- `skills.yaml` supplies registry metadata. Each skill's files are read from
  `skills/<name>/SKILL.md` (path derived from `name`).
- Optional `intel-*` fields drive badges and filters when present.

## Visual style

Colours and type follow intel.com's product pages, sampled from
`/products/details/processors/core-ultra.html` rather than guessed: navy `#000F28`
canvas, white text on slate-blue `#8791A2` secondaries, one saturated blue accent
(`#2D76FF`) for actions and Energy Blue (`#00C7FD`) for links that leave the page,
hairline borders instead of heavy shadows, and fully rounded controls.

**Dark only.** There is no `prefers-color-scheme` branch; the palette is a single set of
tokens on `:root`. `color-scheme: dark` is declared so the browser renders form controls
in their dark variants rather than handing back a white search box.

Headings are IntelOne Display at weight 500 with `-0.01em` tracking, body copy is
IntelOne Text. There is no bold cut of IntelOne Text, so 500 is the heaviest body weight
— `strong` is pinned to 500 to avoid a synthesised bold.

The faces are self-hosted from `public/fonts/`, downloaded from
`https://www.intel.com/intel-shared-assets/fonts/en/`. Only the three weights actually
used are shipped (Text 400/500, Display 500), about 90 KB total. Refresh them with:

```sh
base=https://www.intel.com/intel-shared-assets/fonts/en
for f in intelone-text-regular intelone-text-medium intelone-display-medium; do
  curl -sfS -o "public/fonts/$f.woff2" "$base/$f.woff2"
done
```

## Requirements

Astro 7 needs Node >= 22.12. This repo pins the version in `.nvmrc`:

```sh
nvm use
npm install
```

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Dev server on http://localhost:4321 (status badges off by default) |
| `npm run dev:with-status` | Same, with status badges and the Status filter |
| `npm run build` | Static build to `dist/` (status off by default) |
| `npm run build:with-status` | Static build including status badges and filter |
| `npm run preview` | Serve the built output |
| `npm run check` | Astro + TypeScript diagnostics |

Set `SHOW_SKILL_STATUS=true` (or `1` / `yes`) on any `astro` command to include status; default is off.
