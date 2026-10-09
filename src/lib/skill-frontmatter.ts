import matter from 'gray-matter';

export interface SkillFrontmatter {
  description: string;
  license?: string;
}

function trimString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function isYamlParseError(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false;
  const name = 'name' in error ? String((error as { name: string }).name) : '';
  const message = error instanceof Error ? error.message : String(error);
  return (
    name === 'YAMLException' ||
    message.includes('YAMLException') ||
    message.includes('incomplete explicit mapping pair')
  );
}

/** Frontmatter body between the opening and closing `---` lines. */
function frontmatterRegion(content: string): string | null {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  return match?.[1] ?? null;
}

/**
 * Catalog skills usually use one long `description:` line. Unquoted colons in
 * the prose break js-yaml; wrapping the value in quotes fixes most cases.
 */
function quoteUnquotedDescriptionLine(region: string): string {
  return region
    .split(/\r?\n/)
    .map((line) => {
      const match = line.match(/^description:\s+(.*)$/);
      if (!match) return line;

      const value = match[1]!;
      if (/^["'|>]/.test(value)) return line;

      const escaped = value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
      return `description: "${escaped}"`;
    })
    .join('\n');
}

/** Last resort when YAML still fails: read `description` / `license` as plain lines. */
function readSingleLineFrontmatterFields(region: string): SkillFrontmatter | null {
  let description: string | undefined;
  let license: string | undefined;

  for (const line of region.split(/\r?\n/)) {
    const descriptionMatch = line.match(/^description:\s*(.*)$/);
    if (descriptionMatch) {
      description = descriptionMatch[1]!.trim();
      continue;
    }
    const licenseMatch = line.match(/^license:\s*(.*)$/);
    if (licenseMatch) {
      license = licenseMatch[1]!.trim();
    }
  }

  if (description === undefined) return null;
  return { description, license: license || undefined };
}

function parseMatter(content: string): SkillFrontmatter {
  const { data } = matter(content);
  return {
    description: trimString(data.description),
    license: trimString(data.license) || undefined,
  };
}

/**
 * Parses SKILL.md YAML frontmatter. Tolerates a single-line `description` whose
 * unquoted colons would otherwise make js-yaml fail.
 */
export function parseSkillFrontmatter(
  content: string,
  skillName: string,
  skillMdPath: string,
): SkillFrontmatter {
  try {
    return parseMatter(content);
  } catch (error) {
    if (!isYamlParseError(error)) throw error;

    const region = frontmatterRegion(content);
    if (!region) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(
        `[skills] ${skillName}: invalid SKILL.md frontmatter in ${skillMdPath}: ${detail}`,
      );
    }

    try {
      const repaired = `---\n${quoteUnquotedDescriptionLine(region)}\n---\n`;
      const parsed = parseMatter(repaired);
      console.warn(
        `[skills] ${skillName}: quoted \`description\` for YAML (${skillMdPath}) — unescaped \`:\` in the value`,
      );
      return parsed;
    } catch (repairError) {
      const fallback = readSingleLineFrontmatterFields(region);
      if (fallback) {
        console.warn(
          `[skills] ${skillName}: line fallback for frontmatter after YAML failed (${skillMdPath})`,
        );
        return fallback;
      }

      const detail =
        repairError instanceof Error ? repairError.message : String(repairError);
      throw new Error(
        `[skills] ${skillName}: cannot parse SKILL.md frontmatter in ${skillMdPath}: ${detail}`,
      );
    }
  }
}
