import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Skill } from './skills';
import { skills } from './skills';
import { bundleVisualFor, type BundleVisual } from './bundle-visuals';

declare const __CATALOG_ROOT__: string;
const CATALOG_ROOT = __CATALOG_ROOT__;

interface SkillsShGrouping {
  title: string;
  description: string;
  skills: string[];
}

interface SkillsShFile {
  groupings?: SkillsShGrouping[];
}

export interface SkillGroup {
  /** Stable id for in-tab navigation and data attributes. */
  id: string;
  title: string;
  description: string;
  /** Skills in catalog order as listed in skills.sh.json. */
  skills: Skill[];
  /** Copy-paste command to install every skill in the group. */
  installCommand: string;
  visual: BundleVisual;
}

function groupInstallCommand(skillNames: string[]): string {
  if (skillNames.length === 0) {
    throw new Error('[groups] install command requires at least one skill');
  }
  return `npx skills add intel/skills --skill ${skillNames.join(' ')}`;
}

function slugify(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function loadSkillGroups(): SkillGroup[] {
  const path = join(CATALOG_ROOT, 'skills.sh.json');
  if (!existsSync(path)) {
    console.warn(`[groups] ${path} not found — Bundles tab will be empty`);
    return [];
  }

  let parsed: SkillsShFile;
  try {
    parsed = JSON.parse(readFileSync(path, 'utf8')) as SkillsShFile;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`[groups] cannot read ${path}: ${detail}`);
  }

  const byName = new Map(skills.map((skill) => [skill.name, skill]));
  const ids = new Set<string>();
  const groups: SkillGroup[] = [];

  for (const grouping of parsed.groupings ?? []) {
    const title = grouping.title?.trim();
    if (!title) continue;

    const resolved: Skill[] = [];
    for (const name of grouping.skills ?? []) {
      const skill = byName.get(name);
      if (!skill) {
        throw new Error(
          `[groups] ${title}: unknown skill "${name}" — not in skills.yaml`,
        );
      }
      resolved.push(skill);
    }

    if (resolved.length === 0) {
      throw new Error(`[groups] ${title}: grouping has no skills`);
    }

    let id = slugify(title);
    if (ids.has(id)) {
      id = `${id}-${groups.length}`;
    }
    ids.add(id);

    groups.push({
      id,
      title,
      description: grouping.description?.trim() ?? '',
      skills: resolved,
      installCommand: groupInstallCommand(resolved.map((skill) => skill.name)),
      visual: bundleVisualFor(id),
    });
  }

  return groups;
}

export const skillGroups = loadSkillGroups();
