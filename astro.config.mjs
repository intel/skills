// @ts-check
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';

const projectRoot = fileURLToPath(new URL('.', import.meta.url));

/**
 * Catalog checkout root (skills.yaml and skills/). Set `CATALOG_ROOT` in CI when
 * the webpage and catalog are separate checkouts; locally use the `intel-skills`
 * symlink or point `CATALOG_ROOT` at a main-branch worktree.
 */
function resolveCatalogRoot() {
  const fromEnv = process.env.CATALOG_ROOT;
  if (fromEnv) {
    return path.resolve(fromEnv);
  }
  return fileURLToPath(new URL('./intel-skills/', projectRoot));
}

const catalogRoot = resolveCatalogRoot();

/** Parse SHOW_SKILL_STATUS for build/dev (default: omit status badges and filter). */
function envFlag(name, defaultValue = false) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return defaultValue;
  const normalized = raw.toLowerCase();
  if (normalized === '0' || normalized === 'false' || normalized === 'no') return false;
  if (normalized === '1' || normalized === 'true' || normalized === 'yes') return true;
  return defaultValue;
}

const showSkillStatus = envFlag('SHOW_SKILL_STATUS');

/** URL path the site is served under. Override with SITE_BASE=/ for domain root. */
function normalizeBase(raw) {
  if (!raw || raw === '/') return '/';
  const withLeading = raw.startsWith('/') ? raw : `/${raw}`;
  return withLeading.endsWith('/') ? withLeading : `${withLeading}/`;
}

const siteBase = normalizeBase(process.env.SITE_BASE ?? '/');

export default defineConfig({
  base: siteBase,
  // Static build: the skill catalog is read from the checkout at build time.
  output: 'static',
  vite: {
    define: {
      __CATALOG_ROOT__: JSON.stringify(catalogRoot),
      __SHOW_SKILL_STATUS__: showSkillStatus,
    },
  },
});
