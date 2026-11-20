#!/usr/bin/env bash
#
# Profile-guided optimization (PGO): a release build in two passes, with a
# training run in between.
#
#   scripts/pgo-build.sh instrument --bin agave-validator   # pass 1
#   <run the instrumented binary on a workload like yours, then stop it>
#   scripts/pgo-build.sh build --bin agave-validator        # pass 2
#
# Any arguments after the subcommand are passed to `cargo build --release`, for
# example `-p solana-poh-bench`, `--bin agave-validator` or
# `--manifest-path dev-bins/Cargo.toml -p agave-ledger-tool`. Cargo runs in the
# repository root, so relative paths are relative to it.
#
# The passes are separate commands because you run the training workload in
# between. A profile helps only if it was recorded on a workload like the one
# the binary will run. A profile recorded on a different workload can make the
# binary slower than a normal build.
#
# An instrumented binary writes its profile when it exits normally. A process
# killed by a signal writes nothing, so stop a validator with
# `agave-validator --ledger <ledger dir> exit`.
#
# Environment:
#   PGO_PROFILE_DIR         where .profraw files and the merged profile go
#                           (default: target/pgo). `instrument` deletes the
#                           *.profraw files and merged.profdata in it, and
#                           nothing else.
#   PGO_TARGET              target triple (default: the host's)
#   PGO_INSTRUMENT_LDFLAGS  extra RUSTFLAGS for pass 1 only (default: use lld on
#                           aarch64-unknown-linux-*, see below)
#   RUSTFLAGS               kept and extended in both passes; must be identical
#                           between them apart from the profile flags
set -euo pipefail

usage() {
  echo "usage: $0 {instrument|build} [cargo build args...]" >&2
  exit 1
}
[[ $# -ge 1 ]] || usage
cmd=$1
shift
[[ $cmd == instrument || $cmd == build ]] || usage

cd "$(dirname "$0")/.."

PGO_PROFILE_DIR=${PGO_PROFILE_DIR:-target/pgo}
mkdir -p "$PGO_PROFILE_DIR"
profile_dir=$(cd "$PGO_PROFILE_DIR" && pwd)
profdata="$profile_dir/merged.profdata"

host=$(rustc -vV | awk '/^host: /{print $2}')

# Always pass --target. Without it, RUSTFLAGS also applies to build scripts and
# proc macros, so cargo would build them with profiling on and run them during
# the build. Because of --target, the binaries are written to
# <target dir>/<triple>/release/ instead of <target dir>/release/.
PGO_TARGET=${PGO_TARGET:-$host}

# Setting RUSTFLAGS replaces the rustflags from .cargo/config.toml, which set
# -Ctarget-cpu=x86-64-v2 for x86_64 Linux. If RUSTFLAGS is not set, add that
# flag back so that the PGO build targets the same CPUs as a normal build.
if [[ -z ${RUSTFLAGS:-} && $PGO_TARGET == x86_64-unknown-linux-gnu ]]; then
  RUSTFLAGS="-Ctarget-cpu=x86-64-v2"
fi
base_rustflags=${RUSTFLAGS:-}

# On aarch64 Linux, GNU ld can fail to link the instrumented validator with
# "relocation truncated to fit: R_AARCH64_CALL26", because the binary is larger
# than the +/-128 MB range of a direct call. lld handles this. Only pass 1 uses
# lld, so the optimized binary is linked the same way as a normal build.
# aarch64-apple-darwin links with ld64 and is not affected.
if [[ -z ${PGO_INSTRUMENT_LDFLAGS+set} ]]; then
  PGO_INSTRUMENT_LDFLAGS=""
  if [[ $PGO_TARGET == aarch64-unknown-linux-* ]]; then
    PGO_INSTRUMENT_LDFLAGS="-Clink-arg=-fuse-ld=lld"
    if [[ $cmd == instrument ]] && ! command -v ld.lld >/dev/null; then
      echo "ld.lld not found; install lld, or set PGO_INSTRUMENT_LDFLAGS= to link with the default linker" >&2
      exit 1
    fi
  fi
fi

llvm_profdata="$(rustc --print sysroot)/lib/rustlib/$host/bin/llvm-profdata"
if [[ ! -x $llvm_profdata ]]; then
  echo "llvm-profdata not found; run in $PWD: rustup component add llvm-tools-preview" >&2
  exit 1
fi

if ! command -v jq >/dev/null; then
  echo "jq not found; install jq" >&2
  exit 1
fi

# Build with the given profile flags added to RUSTFLAGS, and print the paths of
# the binaries that cargo built. The paths depend on the workspace: for
# --manifest-path dev-bins/Cargo.toml they are in dev-bins/target/.
cargo_build() {
  local profile_flags=$1
  shift
  RUSTFLAGS="$base_rustflags $profile_flags" \
    cargo build --release --target "$PGO_TARGET" --message-format=json-render-diagnostics "$@" |
    jq -r 'select(.reason == "compiler-artifact" and .executable != null) | "  " + .executable'
}

case $cmd in
  instrument)
    # Remove the profiles of an earlier training run, so that they are not mixed
    # into the new one. Delete only these files, never the directory: it may
    # hold other files.
    rm -f "$profile_dir"/*.profraw "$profdata"
    binaries=$(cargo_build "-Cprofile-generate=$profile_dir $PGO_INSTRUMENT_LDFLAGS" "$@")
    echo
    echo "Instrumented binaries:"
    echo "$binaries"
    echo "Run them on a representative workload and let them exit normally, then run:"
    echo "  $0 build $*"
    ;;
  build)
    if ! compgen -G "$profile_dir/*.profraw" >/dev/null; then
      echo "no .profraw files in $profile_dir" >&2
      echo "run '$0 instrument ...' and exercise the instrumented binary first" >&2
      exit 1
    fi
    "$llvm_profdata" merge -o "$profdata" "$profile_dir"/*.profraw
    binaries=$(cargo_build "-Cprofile-use=$profdata" "$@")
    echo
    echo "Optimized binaries:"
    echo "$binaries"
    ;;
esac
