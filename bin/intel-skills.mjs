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

const LIST_FLAGS = ["--json", "--imported", "--own"];
const VALUE_FLAGS = new Set(["--target", "--dir", "--path"]);

const NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function skillDir(name) {
  if (!NAME.test(String(name))) throw new Error(`invalid skill name: ${name}`);
  const src = path.join(SKILLS_DIR, name);
  if (!fs.existsSync(path.join(src, "SKILL.md"))) {
    // A half-remembered name is the usual cause, so point at the catalog entries that
    // contain it rather than leave the user to run `list` and scan thirty-odd rows.
    const similar = catalog().filter((candidate) => candidate.includes(name));
    const hint = similar.length > 0 ? `; did you mean: ${similar.join(", ")}?` : "";
    throw new Error(`skill not found: ${name}${hint}`);
  }
  return src;
}

/** A one-line YAML scalar without its surrounding quotes, so `list` shows the text alone. */
function unquote(value) {
  if (value.length >= 2 && value.startsWith('"') && value.endsWith('"')) {
    return value.slice(1, -1).replace(/\\(["\\])/g, "$1");
  }
  if (value.length >= 2 && value.startsWith("'") && value.endsWith("'")) {
    return value.slice(1, -1).replace(/''/g, "'");
  }
  return value;
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
  if (inline && !/^[>|]-?$/.test(inline)) return unquote(inline);
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
  const segments = Object.hasOwn(INSTALL_TARGETS, target) ? INSTALL_TARGETS[target] : null;
  // A misspelt --target is an error even when --dir wins, so a typo is not silently ignored.
  if (!segments && (target || !dirOverride)) {
    throw new Error(`unknown target: ${target} (expected one of ${Object.keys(INSTALL_TARGETS).join(", ")})`);
  }
  if (dirOverride) return path.resolve(dirOverride);
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
function install(name, root, force, dryRun = false) {
  const src = skillDir(name);
  const dest = destinationFor(root, name, force);
  const listed = files(src);
  if (dryRun) {
    console.log(`would install ${name} -> ${dest} (${listed.length} file(s))`);
    for (const rel of listed) console.log(`  ${rel}`);
    return;
  }
  fs.mkdirSync(root, { recursive: true });
  fs.rmSync(dest, { recursive: true, force: true });
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
function verify(name, dir, allProblems = false) {
  const src = skillDir(name);
  const installed = path.resolve(dir);
  if (!fs.existsSync(installed)) throw new Error(`no such directory: ${installed}`);
  // A file here used to surface as a raw "ENOTDIR: not a directory, scandir" from node.
  if (!fs.statSync(installed).isDirectory()) throw new Error(`not a directory: ${installed}`);
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
    const shown = allProblems ? problems : problems.slice(0, 4);
    throw new Error(
      `${name} at ${installed} does not match skills/${name}/: ${shown.join("; ")}` +
        (problems.length > shown.length ? ` (+${problems.length - shown.length} more; --all-problems lists them)` : ""),
    );
  }
  console.log(`ok ${name} (${wanted.length} file(s) match skills/${name}/)`);
}

/**
 * Remove an installed skill from a target root.
 *
 * Deliberately narrow: the name must be in the catalog (so a typo or another hub's skill
 * cannot delete an unrelated folder) unless `force` is set, for a skill since dropped from
 * it; the directory must hold a SKILL.md, and it must stay inside the root. It never
 * touches anything but `<root>/<name>`.
 */
function uninstall(name, root, force = false) {
  if (!NAME.test(String(name))) throw new Error(`invalid skill name: ${name}`);
  const inCatalog = fs.existsSync(path.join(SKILLS_DIR, name, "SKILL.md"));
  const dest = path.join(root, name);
  if (!path.resolve(dest).startsWith(`${path.resolve(root)}${path.sep}`)) {
    throw new Error("uninstall path escapes target directory");
  }
  if (!fs.existsSync(path.join(dest, "SKILL.md"))) {
    if (!inCatalog) skillDir(name);
    throw new Error(`not an installed skill: ${dest}`);
  }
  if (!inCatalog && !force) {
    throw new Error(`skill not found: ${name} is not in the catalog; pass --force to remove ${dest} anyway`);
  }
  fs.rmSync(dest, { recursive: true, force: true });
  console.log(`removed ${name} from ${dest}`);
}

function argValue(args, flag) {
  const index = args.indexOf(flag);
  if (index === -1) return null;
  const value = args[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${flag} requires a value`);
  return value;
}

/**
 * Refuse an option the command does not take. A misspelt `--forse` used to be dropped
 * silently, so the command ran with its default and the user never learned the flag
 * had not applied.
 */
function rejectUnknownOptions(args, valueFlags, switches) {
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (valueFlags.includes(arg)) i += 1;
    else if (arg.startsWith("-") && !switches.includes(arg)) throw new Error(`unknown option: ${arg}`);
  }
}

/** The first argument that is neither a flag nor the value of one, so flags may precede it. */
function positionals(args) {
  const found = [];
  for (let i = 0; i < args.length; i += 1) {
    if (VALUE_FLAGS.has(args[i])) i += 1;
    else if (!args[i].startsWith("--") && !found.includes(args[i])) found.push(args[i]);
  }
  return found;
}

function usage(code = 0) {
  console.log(`Usage:
  intel-skills list [--json] [--imported|--own]
  intel-skills show <skill> [--files|--json]
  intel-skills install <skill>...|--all --target ${Object.keys(INSTALL_TARGETS).join("|")} [--dir <path>] [--force] [--dry-run]
  intel-skills uninstall <skill> --target ${Object.keys(INSTALL_TARGETS).join("|")} [--dir <path>] [--force]
  intel-skills verify <skill> --path <installed-skill-dir> [--all-problems]
  intel-skills verify <skill> --target ${Object.keys(INSTALL_TARGETS).join("|")} [--dir <path>]
  intel-skills --version
  intel-skills verify --all --path <skills-dir>

Every command is a local operation on this checkout: no network, and nothing but node
is required. \`list\` marks which skills are copies of another repository's work, and
\`show\` says which commit a copy was taken from.`);
  process.exit(code);
}

const [, , command, ...args] = process.argv;

try {
  if (!command || command === "-h" || command === "--help") usage(0);
  // `install --help` is what a user types to learn the flags; without this it fell through
  // to "invalid skill name: --help" or an exit-1 usage, and `list --help` printed the catalog.
  // A flag's value is never a help request, so `--dir -h` still means a directory named -h.
  if (args.some((arg, i) => (arg === "-h" || arg === "--help") && !VALUE_FLAGS.has(args[i - 1]))) {
    usage(0);
  }

  if (command === "--version" || command === "-v") {
    console.log(JSON.parse(fs.readFileSync(path.join(ROOT, "package.json"), "utf8")).version);
    process.exit(0);
  }

  if (command === "list") {
    const unknown = args.find((arg) => !LIST_FLAGS.includes(arg));
    if (unknown) throw new Error(`unknown option for list: ${unknown}`);
    const json = args.includes("--json");
    const imported = args.includes("--imported");
    const own = args.includes("--own");
    if (imported && own) throw new Error("--imported and --own cannot be combined");
    const rows = [];
    for (const name of catalog()) {
      const origin = source(name);
      if ((imported && !origin) || (own && origin)) continue;
      rows.push({
        name,
        kind: origin ? "imported" : "own",
        repo: origin ? origin.repo : null,
        commit: origin ? String(origin.commit) : null,
        license: origin ? origin.license ?? null : null,
        description: description(name),
      });
    }
    if (json) {
      // The tab-separated form truncates the description to 100 characters and is built
      // for a person; this is the same catalog, whole, for a script that would otherwise
      // have to split on tabs.
      console.log(JSON.stringify(rows, null, 2));
    } else {
      for (const row of rows) {
        const kind = row.repo
          ? `imported\t${row.repo.split("/").slice(-1)[0]}@${row.commit.slice(0, 12)}`
          : "own\t-";
        console.log(`${row.name}\t${kind}\t${row.description.slice(0, 100)}`);
      }
    }
  } else if (command === "show") {
    rejectUnknownOptions(args, [], ["--files", "--json"]);
    const names = positionals(args);
    // Without this, `show` alone reported "invalid skill name: undefined", an error
    // about a value the user never typed. Every other command prints usage instead.
    if (names.length === 0) usage(1);
    if (names.length > 1) throw new Error(`show takes one skill name (got extra argument ${names[1]})`);
    const name = names[0];
    if (args.includes("--files") && args.includes("--json")) throw new Error("--files and --json cannot be combined");
    const src = skillDir(name);
    if (args.includes("--files")) {
      // The exact set `install` would write, so a reviewer can see what lands on disk
      // without installing it.
      for (const rel of files(src)) console.log(rel);
      process.exit(0);
    }
    const origin = source(name);
    if (args.includes("--json")) {
      // What a script reads instead of scraping the SKILL.md dump below.
      console.log(JSON.stringify({ name, description: description(name), source: origin, files: files(src) }, null, 2));
      process.exit(0);
    }
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
    rejectUnknownOptions(args, ["--target", "--dir"], ["--force", "--all", "--dry-run"]);
    if (args.includes("--all") && args.some((arg, i) => !arg.startsWith("-") && !["--target", "--dir"].includes(args[i - 1]))) {
      throw new Error("--all installs every skill; do not also name one");
    }
    const target = argValue(args, "--target");
    const dirOverride = argValue(args, "--dir");
    if (!target && !dirOverride) usage(1);
    const root = targetRoot(target, dirOverride);
    const force = args.includes("--force");
    const dryRun = args.includes("--dry-run");
    // One skill's failure must not decide the others. Aborting on the first error left an
    // arbitrary prefix of the catalog installed and said nothing about the rest, which is
    // the worst of both: partial state, and no list of what is missing. So each is
    // attempted, every failure is reported, and the exit code still says the run did not
    // fully succeed.
    const names = args.includes("--all") ? catalog() : positionals(args);
    if (names.length === 0) usage(1);
    if (names.length === 1 && !args.includes("--all")) {
      install(names[0], root, force, dryRun);
    } else {
      const failures = [];
      for (const name of names) {
        try {
          install(name, root, force, dryRun);
        } catch (error) {
          failures.push(`${name}: ${error.message}`);
          console.error(`error: ${name}: ${error.message}`);
        }
      }
      console.log(`${names.length - failures.length}/${names.length} ${dryRun ? "would be installed to" : "installed to"} ${root}`);
      if (failures.length > 0) {
        console.error(`${failures.length} skill(s) not installed:`);
        for (const failure of failures) console.error(`  ${failure}`);
        process.exit(1);
      }
    }
  } else if (command === "uninstall") {
    rejectUnknownOptions(args, ["--target", "--dir"], ["--force"]);
    const target = argValue(args, "--target");
    const dirOverride = argValue(args, "--dir");
    if (!target && !dirOverride) usage(1);
    const names = positionals(args);
    if (names.length === 0) usage(1);
    if (names.length > 1) throw new Error(`uninstall takes one skill name (got extra argument ${names[1]})`);
    const name = names[0];
    uninstall(name, targetRoot(target, dirOverride), args.includes("--force"));
  } else if (command === "verify") {
    rejectUnknownOptions(args, ["--path", "--target", "--dir"], ["--all", "--all-problems"]);
    const target = argValue(args, "--target");
    const dirOverride = argValue(args, "--dir");
    const explicit = argValue(args, "--path");
    const allProblems = args.includes("--all-problems");
    // `--target` names the place `install` wrote to, so verify can check it without the
    // caller spelling out the per-agent directory.
    if (!explicit && !target && !dirOverride) usage(1);
    const names = positionals(args);
    if (names.length > 1) throw new Error(`verify takes one skill name (got extra argument ${names[1]})`);
    if (args.includes("--all")) {
      if (names.length > 0) throw new Error("--all verifies every skill; do not also name one");
      // The counterpart of `install --all`: the directory it wrote into, with every
      // catalog skill checked under it. Each is attempted and every failure is listed,
      // for the same reason `install --all` does not stop at the first one.
      const root = path.resolve(explicit ?? targetRoot(target, dirOverride));
      if (!fs.existsSync(root)) throw new Error(`no such directory: ${root}`);
      const all = catalog();
      const failures = [];
      for (const name of all) {
        try {
          verify(name, path.join(root, name), allProblems);
        } catch (error) {
          failures.push(`${name}: ${error.message}`);
          console.error(`error: ${name}: ${error.message}`);
        }
      }
      console.log(`${all.length - failures.length}/${all.length} verified under ${root}`);
      if (failures.length > 0) process.exit(1);
    } else {
      const name = names[0];
      if (!name) usage(1);
      verify(name, explicit ?? path.join(targetRoot(target, dirOverride), name), allProblems);
    }
  } else {
    console.error(`error: unknown command: ${command}`);
    usage(1);
  }
} catch (error) {
  console.error(`error: ${error.message}`);
  process.exit(1);
}
