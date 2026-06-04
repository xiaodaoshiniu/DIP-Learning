#!/usr/bin/env bash
set -euo pipefail

cd /home/czr

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

echo "conda envs:"
conda env list || true

echo "all colmap candidates:"
(
  command -v -a colmap || true
  find "$HOME" -maxdepth 5 -type f -name colmap 2>/dev/null || true
) | sort -u

for env_dir in "$HOME"/miniconda3/envs/* "$HOME"/anaconda3/envs/*; do
  [ -d "$env_dir" ] || continue
  if [ -x "$env_dir/bin/colmap" ]; then
    echo "candidate=$env_dir/bin/colmap"
    if [ -e "$env_dir/lib/libstdc++.so.6" ]; then
      echo "libstdc++=$env_dir/lib/libstdc++.so.6"
      strings "$env_dir/lib/libstdc++.so.6" | grep -E 'GLIBCXX_3\.4\.(31|32)|CXXABI_1\.3\.15' | tail -n 5 || true
    fi
    LD_LIBRARY_PATH="$env_dir/lib:${LD_LIBRARY_PATH:-}" "$env_dir/bin/colmap" -h >/tmp/colmap_test.log 2>&1 && echo "works" || tail -n 3 /tmp/colmap_test.log
  fi
done
