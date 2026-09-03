#!/usr/bin/env node
/**
 * `npx github:intel/skills` -- put a skill from this catalog on a machine.
 *
 * Every skill in the catalog is here in full, including the ones another Intel
 * repository maintains: those are vendored copies pinned in `skills.yaml`, each
 * carrying a `.source.json` that says which repository, path and commit it came from.
 * So install is a copy of a directory, and that is the whole design:
 *
 *   1. No network, no `git`, no hash manifest. The bytes an install writes are the
 *      bytes in the checkout, which a reviewer has read and CI has scanned. There is
 *      no install-time fetch left to fail, to be intercepted, or to disagree with
 *      what this repository ships.
 *   2. What lands on disk is what `skills/<name>/` holds, minus the two directories
 *      that are evidence about the skill rather than part of it (see NOT_INSTALLED).
 *      `verify` is that same comparison run against an already-installed copy.
 *
 * Zero dependencies on purpose -- `npx` on a locked-down host should need nothing but
 * node.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SKILLS_DIR = path.join(ROOT, "skills");
const SOURCE_FILE = ".source.json";

/**
 * `codex` is an alias for `agents`, not a directory of its own: Codex documents
 * `$HOME/.agents/skills` and never `~/.codex/skills`, so the obvious-looking path is the
 * one Codex does not read. The alias stays because it is the name a user reaches for.
 */
const INSTALL_TARGETS = {
  "claude-code": [".claude", "skills"],
  codex: [".agents", "skills"],
  agents: [".agents", "skills"],
};

/**
 * Directories a skill keeps in the catalog and does not install.
 *
 * `evals/` is this repository's test fixtures — and an eval case is allowed to name a
 * phrase precisely because a skill must not contain it, so shipping it into an agent's
 * skills directory puts that phrase in the agent's reach. `perf/` is measurement
 * provenance for reviewers. Neither is something the agent loads.
 */
const NOT_INSTALLED = new Set(["evals", "perf"]);

const NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function skillDir(name) {
  if (!NAME.test(String(name))) throw new Error(`invalid skill name: ${name}`);
  const src = path.join(SKILLS_DIR, name);
  if (!fs.existsSync(path.join(src, "SKILL.md"))) throw new Error(`skill not found: ${name}`);
  return src;
}

/** The description an agent routes on, read from the frontmatter of SKILL.md. */
function description(name) {
  const skillMd = path.join(SKILLS_DIR, name, "SKILL.md");
  if (!fs.existsSync(skillMd)) return "";
  const text = fs.readFileSync(skillMd, "utf8").replace(/\r\n/g, "\n");
  const end = text.indexOf("\n---", 4);
  if (!text.startsWith("---\n") || end === -1) return "";
  const lines = text.slice(4, end).split("\n");
  const index = lines.findIndex((line) => /^description:/.test(line));
  if (index === -1) return "";
  const inline = lines[index].replace(/^description:\s*/, "").trim();
  if (inline && !/^[>|]-?$/.test(inline)) return inline;
  const block = [];
  for (let i = index + 1; i < lines.length && /^\s{2,}\S|^\s*$/.test(lines[i]); i += 1) {
    block.push(lines[i].trim());
  }
  return block.join(" ").trim();
}

/** `.source.json` if the skill is a copy of someone else's, otherwise null. */
function source(name) {
  const file = path.join(SKILLS_DIR, name, SOURCE_FILE);
  if (!fs.existsSync(file)) return null;
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    throw new Error(`skills/${name}/${SOURCE_FILE} is not valid JSON: ${error.message}`);
  }
}

function catalog() {
  return fs
    .readdirSync(SKILLS_DIR, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && fs.existsSync(path.join(SKILLS_DIR, entry.name, "SKILL.md")))
    .map((entry) => entry.name)
    .sort();
}

/** Every installable file under a skill, relative and sorted. NOT_INSTALLED is skipped. */
function files(dir) {
  const walk = (base, prefix = "") =>
    fs.readdirSync(base, { withFileTypes: true }).flatMap((item) => {
      const rel = prefix ? `${prefix}/${item.name}` : item.name;
      if (!prefix && NOT_INSTALLED.has(item.name)) return [];
      if (item.isDirectory()) return walk(path.join(base, item.name), rel);
      if (item.isFile()) return [rel];
      throw new Error(`unsupported entry in skill tree: ${rel}`);
    });
  return walk(dir).sort();
}

function targetRoot(target, dirOverride) {
  if (dirOverride) return path.resolve(dirOverride);
  const segments = Object.hasOwn(INSTALL_TARGETS, target) ? INSTALL_TARGETS[target] : null;
  if (!segments) {
    throw new Error(`unknown target: ${target} (expected one of ${Object.keys(INSTALL_TARGETS).join(", ")})`);
  }
  return path.join(os.homedir(), ...segments);
}

function destinationFor(root, name, force) {
  const dest = path.join(root, name);
  if (!path.resolve(dest).startsWith(`${path.resolve(root)}${path.sep}`)) {
    throw new Error("install destination escapes target directory");
  }
  if (fs.existsSync(dest) && !force) {
    throw new Error(`destination already exists: ${dest}; use --force to overwrite`);
  }
  return dest;
}

/**
 * Copy the listed files, keeping each one's mode.
 *
 * The mode matters and is not decoration: a skill whose body says to run
 * `scripts/check_runtime_preflight.sh` ships it executable, and a copy that dropped
 * the bit would fail on the first command the skill documents.
 */
function install(name, root, force) {
  const src = skillDir(name);
  const dest = destinationFor(root, name, force);
  fs.mkdirSync(root, { recursive: true });
  fs.rmSync(dest, { recursive: true, force: true });
  const listed = files(src);
  for (const rel of listed) {
    const from = path.join(src, rel);
    const to = path.join(dest, rel);
    fs.mkdirSync(path.dirname(to), { recursive: true });
    fs.copyFileSync(from, to);
    fs.chmodSync(to, fs.statSync(from).mode & 0o777);
  }
  const origin = source(name);
  const provenance = origin
    ? `${origin.repo} @ ${String(origin.commit).slice(0, 12)}, ${origin.license}`
    : "maintained in this repository";
  console.log(`installed ${name} -> ${dest} (${listed.length} file(s), ${provenance})`);
}

/**
 * Compare an installed skill against the catalog, byte for byte, offline.
 *
 * Both directions, because both are drift a reader would want to know about: a file
 * the catalog has and the installation does not, and a file the installation has that
 * no skill here ships. What this cannot tell you is whether the catalog itself still
 * matches the upstream an imported skill was copied from — that needs the network and
 * is `python3 tools/sync_external.py --check`.
 */
function verify(name, dir) {
  const src = skillDir(name);
  const installed = path.resolve(dir);
  if (!fs.existsSync(installed)) throw new Error(`no such directory: ${installed}`);
  const digest = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");

  const wanted = files(src);
  const present = new Set(files(installed));
  const problems = [];
  for (const rel of wanted) {
    if (!present.has(rel)) {
      problems.push(`${rel} is missing`);
    } else if (digest(path.join(src, rel)) !== digest(path.join(installed, rel))) {
      problems.push(`${rel} differs from the catalog`);
    }
    present.delete(rel);
  }
  for (const rel of [...present].sort()) {
    problems.push(`${rel} is not part of this skill`);
  }
  if (problems.length > 0) {
    throw new Error(
      `${name} at ${installed} does not match skills/${name}/: ${problems.slice(0, 4).join("; ")}` +
        (problems.length > 4 ? ` (+${problems.length - 4} more)` : ""),
    );
  }
  console.log(`ok ${name} (${wanted.length} file(s) match skills/${name}/)`);
}

function argValue(args, flag) {
  const index = args.indexOf(flag);
  if (index === -1) return null;
  const value = args[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${flag} requires a value`);
  return value;
}

function usage(code = 0) {
  console.log(`Usage:
  intel-skills list
  intel-skills show <skill>
  intel-skills install <skill>|--all --target ${Object.keys(INSTALL_TARGETS).join("|")} [--dir <path>] [--force]
  intel-skills verify <skill> --path <installed-skill-dir>

Every command is a local operation on this checkout: no network, and nothing but node
is required. \`list\` marks which skills are copies of another repository's work, and
\`show\` says which commit a copy was taken from.`);
  process.exit(code);
}

const [, , command, ...args] = process.argv;

try {
  if (!command || command === "-h" || command === "--help") usage(0);

  if (command === "list") {
    for (const name of catalog()) {
      const origin = source(name);
      const kind = origin
        ? `imported\t${origin.repo.split("/").slice(-1)[0]}@${String(origin.commit).slice(0, 12)}`
        : "own\t-";
      console.log(`${name}\t${kind}\t${description(name).slice(0, 100)}`);
    }
  } else if (command === "show") {
    const name = args[0];
    const src = skillDir(name);
    const origin = source(name);
    if (origin) {
      console.log(`upstream:  ${origin.repo}/tree/${origin.commit}/${origin.path}`);
      console.log(`pinned:    ${origin.commit}`);
      console.log(`license:   ${origin.license}`);
      if (origin["modified-files"]) {
        console.log(`modified:  ${origin["modified-files"].join(", ")}`);
      }
      console.log("");
    }
    console.log(fs.readFileSync(path.join(src, "SKILL.md"), "utf8"));
  } else if (command === "install") {
    const target = argValue(args, "--target");
    const dirOverride = argValue(args, "--dir");
    if (!target && !dirOverride) usage(1);
    const root = targetRoot(target, dirOverride);
    const force = args.includes("--force");
    if (args.includes("--all")) {
      // One skill's failure must not decide the other thirty-two. Aborting on the
      // first error left an arbitrary prefix of the catalog installed and said nothing
      // about the rest, which is the worst of both: partial state, and no list of what
      // is missing. So each is attempted, every failure is reported, and the exit code
      // still says the run did not fully succeed.
      const names = catalog();
      const failures = [];
      for (const name of names) {
        try {
          install(name, root, force);
        } catch (error) {
          failures.push(`${name}: ${error.message}`);
          console.error(`error: ${name}: ${error.message}`);
        }
      }
      console.log(`${names.length - failures.length}/${names.length} installed to ${root}`);
      if (failures.length > 0) {
        console.error(`${failures.length} skill(s) not installed:`);
        for (const failure of failures) console.error(`  ${failure}`);
        process.exit(1);
      }
    } else {
      const name = args[0];
      if (!name || name.startsWith("--")) usage(1);
      install(name, root, force);
    }
  } else if (command === "verify") {
    const name = args[0];
    const dir = argValue(args, "--path");
    if (!name || !dir) usage(1);
    verify(name, dir);
  } else {
    usage(1);
  }
} catch (error) {
  console.error(`error: ${error.message}`);
  process.exit(1);
}
