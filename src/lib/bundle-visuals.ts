/** Decorative icon and accent for a skills.sh bundle card. */
export type BundleIconVariant =
  | 'gpu-host'
  | 'migration'
  | 'deploy'
  | 'vllm'
  | 'pytorch'
  | 'sglang'
  | 'llamacpp'
  | 'profile'
  | 'dpnp'
  | 'cpu-python'
  | 'perf'
  | 'bundle';

export type BundleAccent = 'blue' | 'violet' | 'cyan';

export interface BundleVisual {
  icon: BundleIconVariant;
  accent: BundleAccent;
}

const VISUAL_BY_ID: Record<string, BundleVisual> = {
  'intel-gpu-xpu-host-setup': { icon: 'gpu-host', accent: 'blue' },
  'cuda-to-intel-xpu-migration': { icon: 'migration', accent: 'violet' },
  'model-planning-deployment': { icon: 'deploy', accent: 'cyan' },
  'vllm-serving-on-intel-gpu': { icon: 'vllm', accent: 'blue' },
  'pytorch-on-intel-gpu': { icon: 'pytorch', accent: 'violet' },
  'sglang-on-intel-gpu': { icon: 'sglang', accent: 'cyan' },
  'llama-cpp-gguf-on-intel-gpu': { icon: 'llamacpp', accent: 'blue' },
  'sycl-level-zero-profiling': { icon: 'profile', accent: 'violet' },
  'intel-data-parallel-numpy-dpnp': { icon: 'dpnp', accent: 'cyan' },
  'cpu-python-acceleration': { icon: 'cpu-python', accent: 'blue' },
  'linux-cpu-performance-benchmarking': { icon: 'perf', accent: 'violet' },
  'other-skills': { icon: 'bundle', accent: 'cyan' },
};

const DEFAULT_VISUAL: BundleVisual = { icon: 'bundle', accent: 'blue' };

export function bundleVisualFor(id: string): BundleVisual {
  return VISUAL_BY_ID[id] ?? DEFAULT_VISUAL;
}
