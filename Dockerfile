FROM rust:1-alpine AS build
ARG FINAL_CPU=cascadelake
RUN apk add --no-cache musl-dev binutils && rustup component add llvm-tools-preview
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY src src
COPY gen gen
RUN mkdir -p /out && \
    cargo build --release --manifest-path gen/Cargo.toml && \
    cp /src/gen/target/release/generate /out/generate && \
    if /out/generate --seed 77 --events 300000 --clicks 75000 --output /tmp/pgo-data --threads 4 && \
        mkdir -p /tmp/pgo /tmp/pgo-out && \
        RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+crt-static -C relocation-model=static -C link-arg=-no-pie -C force-unwind-tables=no -C link-arg=-Wl,--gc-sections -C llvm-args=-x86-branches-within-32B-boundaries --cfg pgo_gen -C profile-generate=/tmp/pgo" \
        cargo build --release --locked --bin sessions && \
        SS_THREADS=8 /src/target/release/sessions /tmp/pgo-data /tmp/pgo-out && \
        "$(find "$(rustc --print sysroot)" -name "llvm-profdata" | head -n1)" merge -o /tmp/pgo/merged.profdata /tmp/pgo/*.profraw; then \
        echo "PGO training succeeded"; \
    else \
        echo "PGO training failed, building without profile" >&2; \
        rm -f /tmp/pgo/merged.profdata; \
    fi && \
    rm -rf /tmp/pgo-data /tmp/pgo-out /tmp/pgo/*.profraw
RUN PGO_FLAG=""; \
    if [ -s /tmp/pgo/merged.profdata ]; then PGO_FLAG="-C profile-use=/tmp/pgo/merged.profdata"; fi; \
    RUSTFLAGS="-C target-cpu=${FINAL_CPU} -C target-feature=+crt-static -C relocation-model=static -C link-arg=-no-pie -C force-unwind-tables=no -C link-arg=-Wl,--gc-sections -C llvm-args=-x86-branches-within-32B-boundaries -C llvm-args=-x86-pad-max-prefix-size=5 ${PGO_FLAG}" \
    cargo build --release --locked --bin sessions && \
    cp /src/target/release/sessions /out/sessions && \
    RUSTFLAGS="-C target-cpu=${FINAL_CPU} -C target-feature=+crt-static,-avx512f -C relocation-model=static -C link-arg=-no-pie -C force-unwind-tables=no -C link-arg=-Wl,--gc-sections -C llvm-args=-x86-branches-within-32B-boundaries -C llvm-args=-x86-pad-max-prefix-size=5 ${PGO_FLAG}" \
    cargo build --release --locked --bin sessions && \
    cp /src/target/release/sessions /out/sessions_nozmm && \
    RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+crt-static -C relocation-model=static -C link-arg=-no-pie -C force-unwind-tables=no -C link-arg=-Wl,--gc-sections" \
    cargo build --release --locked --bin probe && \
    cp /src/target/release/probe /out/probe && \
    strip --strip-all -R .eh_frame -R .eh_frame_hdr -R .gcc_except_table -R .comment -R .note.gnu.build-id /out/sessions /out/sessions_nozmm /out/probe /out/generate

FROM scratch
COPY --from=build /out/sessions /sessions
COPY --from=build /out/sessions_nozmm /sessions_nozmm
COPY --from=build /out/probe /probe
COPY --from=build /out/generate /generate
USER 1000:1000
CMD ["/probe"]
