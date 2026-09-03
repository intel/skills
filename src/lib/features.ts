/** Build-time flags from astro.config.mjs (`vite.define`). */
declare const __SHOW_SKILL_STATUS__: boolean;

/** When false (default), status badges and the status filter row are omitted from the build. */
export const showSkillStatus = __SHOW_SKILL_STATUS__;
