# Official sources for this skill

Where the claims in `SKILL.md` come from. Load this when the question is the current
name or signature of an algorithm, the deprecation status of a `tbb::` spelling, or
how the package is found by CMake in the installed layout.

## oneTBB

| Source | Use it for |
|---|---|
| [oneTBB documentation](https://uxlfoundation.github.io/oneTBB/) | entry point: the algorithms, ranges, containers, and the namespace they live in |
| [uxlfoundation/oneTBB](https://github.com/uxlfoundation/oneTBB) | source and release notes. **The authority on what a name is called now**: oneTBB renamed and removed parts of the pre-oneAPI API, and some old spellings still compile in builds that ship a compatibility layer, so code that builds is not proof the name is current |
| [Intel oneAPI Threading Building Blocks](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html) | how the library is distributed and which environment script puts its CMake config on the search path |
| [oneAPI specification](https://uxlfoundation.github.io/oneAPI-spec/) | the specified behaviour of the parallel algorithms, when the question is the contract rather than one implementation |

## The C++ side of the answer

| Source | Use it for |
|---|---|
| [CMake `find_package`](https://cmake.org/cmake/help/latest/command/find_package.html) | how `find_package(TBB REQUIRED)` searches, and what `CMAKE_PREFIX_PATH` changes — the fix for the build that cannot find the config |

## How to use these

- **Do not answer an API question from memory.** The oneAPI transition renamed
  things, and a plausible-looking `tbb::` name can be the deprecated one. Check the
  documentation for the installed version.
- **Grain size and partitioner questions need a measurement, not a page.** These
  sources describe what the knobs are; only the user's own timing says what to set
  them to, and `SKILL.md` deliberately advises leaving them alone first.
- **Nothing here is a source of speedup figures.** A parallel loop is not
  automatically faster, and the numbers on a vendor page describe someone else's
  machine.
