/** Decorative icon and accent for a skills.sh bundle card. */
export type BundleIconVariant = 'deploy' | 'dpnp' | 'perf' | 'bundle';

export type BundleAccent = 'blue' | 'violet' | 'cyan';

export interface BundleVisual {
  icon: BundleIconVariant;
  accent: BundleAccent;
}

/**
 * Keys are slugified `skills.sh.json` grouping titles (see groups.ts).
 * Keep this map in sync with groupings that ship on main.
 */
const VISUAL_BY_ID: Record<string, BundleVisual> = {
  'deploy-models-on-intel-gpus': { icon: 'deploy', accent: 'cyan' },
  'intel-data-parallel-numpy-dpnp': { icon: 'dpnp', accent: 'blue' },
  'linux-cpu-performance-benchmarking': { icon: 'perf', accent: 'violet' },
};

const DEFAULT_VISUAL: BundleVisual = { icon: 'bundle', accent: 'blue' };

export function bundleVisualFor(id: string): BundleVisual {
  return VISUAL_BY_ID[id] ?? DEFAULT_VISUAL;
}
