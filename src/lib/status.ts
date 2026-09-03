/**
 * How the `status` field in skills.yaml is presented on the page.
 *
 * ── This is the only place the raw statuses are named. ──
 *
 * To relabel, edit the `label` strings below and nothing else. Cards, the
 * detail dialog, the filter chips and the search index all read the label from
 * here, so renaming `Experimental` to `Alpha` renames it everywhere, including
 * the facet the reader filters by.
 *
 *   draft       -> Experimental
 *   incubating  -> Preview
 *   published   -> Preview   (default in skills.yaml when the field is omitted)
 *   validated   -> Pre-release
 *   promoted    -> Release
 *
 * `tone` is deliberately separate from `label`: it names the maturity step
 * rather than the wording, so a relabel to alpha/beta/release keeps its colours
 * without touching CSS. The colours themselves are aliased in global.css.
 */

export type StatusTone = 'experimental' | 'preview' | 'prerelease' | 'release' | 'unknown';

export interface StatusPresentation {
  label: string;
  tone: StatusTone;
}

export const STATUS_PRESENTATION: Record<string, StatusPresentation> = {
  draft: { label: 'Experimental', tone: 'experimental' },
  incubating: { label: 'Preview', tone: 'preview' },
  published: { label: 'Preview', tone: 'preview' },
  validated: { label: 'Pre-release', tone: 'prerelease' },
  promoted: { label: 'Release', tone: 'release' },
};

/**
 * An unmapped status is shown verbatim rather than hidden, so a status added to
 * skills.yaml before this file catches up is visible instead of silently
 * dropped. The build warns so it does not stay that way.
 */
export function presentStatus(status: string): StatusPresentation {
  const known = STATUS_PRESENTATION[status];
  if (known) return known;

  console.warn(
    `[skills] status "${status}" has no entry in src/lib/status.ts; ` +
      'showing it verbatim. Add a label and tone for it.',
  );
  return { label: status, tone: 'unknown' };
}
