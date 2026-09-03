import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import matter from 'gray-matter';
import { parse as parseYaml } from 'yaml';
import { presentStatus, type StatusTone } from './status';
import { showSkillStatus } from './features';

/** The intel/skills checkout. Injected by astro.config.mjs, see the note there. */
declare const __CATALOG_ROOT__: string;
const CATALOG_ROOT = __CATALOG_ROOT__;

/** Canonical repo the catalog is published from, used for non-external skills. */
const CATALOG_REPO = 'https://github.com/intel/skills';

/** One entry of skills.yaml, before it is joined with its SKILL.md. */
interface RegistryEntry {
  name?: string;
  maintainer?: string;
  status?: string;
  'intel-products'?: string;
  'intel-hw-class'?: string;
  'intel-hw-validated-on'?: string;
  'intel-workload-models'?: string;
  'intel-source-ledger'?: string;
  'external-repo'?: string;
  'external-commit'?: string;
  'external-path'?: string;
  'external-license'?: string;
}

export interface Skill {
  name: string;
  /** Human-readable title derived from the skill slug. */
  title: string;
  /** Full description from SKILL.md frontmatter. */
  description: string;
  /** Lead sentence of the description, for the card. */
  summary: string;
  license?: string;
  /** Raw status from skills.yaml: draft | incubating | validated | promoted. */
  status: string;
  /** Reader-facing wording for `status`, from src/lib/status.ts. */
  statusLabel: string;
  /** Maturity step the status badge is coloured by. */
  statusTone: StatusTone;
  maintainer: string;
  /** Intel product names this skill covers. */
  products: string[];
  /** cpu | gpu | npu | accelerator, or empty when it does not apply. */
  hwClass: string;
  /** Specific SKUs the skill was validated on. */
  validatedOn: string[];
  /** Command that installs the skill. */
  installCommand: string;
  /** GitHub URL of the skill's original source. */
  sourceUrl: string;
  /** Lowercased haystack the client-side search matches against. */
  searchText: string;
}

/** One filterable badge value, with how many skills carry it. */
export interface Facet {
  value: string;
  count: number;
}

/**
 * Splits a delimited catalog field. An empty value means "does not apply",
 * so it yields no badges rather than one blank badge.
 */
function splitField(value: string | undefined, separator: string): string[] {
  if (!value) return [];
  return value
    .split(separator)
    .map((part) => part.trim())
    .filter(Boolean);
}

/** A summary shorter than this reads as a fragment, so it absorbs the next sentence. */
const MIN_SUMMARY_LENGTH = 40;

/**
 * Takes the lead sentence of a description. Descriptions in this catalog open
 * with a one-line summary and then continue into trigger and scope guidance,
 * which is more than a card can show.
 *
 * A boundary is a terminator followed by whitespace and a capital letter (or the
 * end of the text). Requiring both keeps `llama.cpp` and `e.g.` from being read
 * as the end of the summary.
 */
function leadSentence(description: string): string {
  const text = description.trim();
  if (!text) return '';

  const boundary = /[.!?](?=\s+["'“(]?[A-Z]|\s*$)/g;
  let match: RegExpExecArray | null;
  while ((match = boundary.exec(text)) !== null) {
    const end = match.index + 1;
    if (end >= MIN_SUMMARY_LENGTH) return text.slice(0, end);
  }
  return text;
}

/** Turns a skill slug into a display title: `dpnp-linalg-fft` -> `dpnp linalg fft`. */
function toTitle(name: string): string {
  return name.replace(/-/g, ' ');
}

/** Turns a registry string field into a trimmed string, or "" when absent. */
function optionalString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

/** Directory under the catalog checkout where a skill's files live: skills/<name>. */
function catalogSkillDir(name: string): string {
  return join('skills', name);
}

/**
 * Resolves the GitHub URL of the skill's original source. External skills point
 * at the upstream repo pinned to the commit the body was vendored from; the rest
 * point at their directory in the catalog repo.
 */
function sourceUrlFor(name: string, entry: RegistryEntry): string {
  const repo = optionalString(entry['external-repo']);
  const commit = optionalString(entry['external-commit']);
  const externalPath = optionalString(entry['external-path']);

  if (repo && commit && externalPath) {
    return `${repo.replace(/\/$/, '')}/tree/${commit}/${externalPath}`;
  }
  return `${CATALOG_REPO}/tree/main/${catalogSkillDir(name).replace(/\\/g, '/')}`;
}

function loadSkills(): Skill[] {
  const parsed = parseYaml(readFileSync(join(CATALOG_ROOT, 'skills.yaml'), 'utf8'));
  if (!Array.isArray(parsed)) {
    throw new Error('[skills] skills.yaml must be a YAML list of skill entries');
  }

  const skills: Skill[] = [];

  for (const raw of parsed) {
    if (!raw || typeof raw !== 'object') {
      console.warn('[skills] skipping non-object registry entry');
      continue;
    }

    const entry = raw as RegistryEntry;
    const name = optionalString(entry.name);
    if (!name) {
      console.warn('[skills] skipping registry entry without name');
      continue;
    }

    const skillMdPath = join(CATALOG_ROOT, catalogSkillDir(name), 'SKILL.md');
    let skillMd: string;
    try {
      skillMd = readFileSync(skillMdPath, 'utf8');
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(`[skills] ${name}: cannot read ${skillMdPath}: ${detail}`);
    }

    const frontmatter = matter(skillMd).data as {
      description?: unknown;
      license?: unknown;
    };
    const description = optionalString(frontmatter.description);
    const summary = leadSentence(description);
    const products = splitField(optionalString(entry['intel-products']), ',');
    const hwClass = optionalString(entry['intel-hw-class']);
    const validatedOn = splitField(optionalString(entry['intel-hw-validated-on']), ';');
    const rawStatus = optionalString(entry.status);
    const catalogStatus = rawStatus || 'published';
    const status = presentStatus(catalogStatus);

    skills.push({
      name,
      title: toTitle(name),
      description,
      summary,
      license: optionalString(frontmatter.license) || undefined,
      status: catalogStatus,
      statusLabel: status.label,
      statusTone: status.tone,
      maintainer: optionalString(entry.maintainer),
      products,
      hwClass,
      validatedOn,
      installCommand: `npx skills add intel/skills --skill ${name}`,
      sourceUrl: sourceUrlFor(name, entry),
      // Summary rather than the full description: the rest of a description
      // is trigger and scope guidance that names sibling skills ("use
      // vllm-xpu-run instead"), and matching on that made a search for a
      // runtime return every skill that merely cross-references it. Badge
      // values are included so "arc" or "gpu" finds skills whose prose never
      // spells the product out.
      searchText: [
        name,
        summary,
        ...(showSkillStatus ? [status.label] : []),
        hwClass,
        ...products,
        ...validatedOn,
      ]
        .join(' ')
        .toLowerCase(),
    });
  }

  return skills.sort((a, b) => a.name.localeCompare(b.name));
}

/** Counts distinct values of one badge dimension, most common first. */
function facetsOf(values: (skill: Skill) => string[]): Facet[] {
  const counts = new Map<string, number>();
  for (const skill of skills) {
    for (const value of values(skill)) {
      counts.set(value, (counts.get(value) ?? 0) + 1);
    }
  }
  return [...counts]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => b.count - a.count || a.value.localeCompare(b.value));
}

export const skills = loadSkills();

export const statusFacets = facetsOf((skill) => [skill.statusLabel]);
export const hwClassFacets = facetsOf((skill) => (skill.hwClass ? [skill.hwClass] : []));
export const productFacets = facetsOf((skill) => skill.products);
export const validatedOnFacets = facetsOf((skill) => skill.validatedOn);
