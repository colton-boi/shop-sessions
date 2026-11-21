FROM rust:1-alpine AS build
RUN apk add --no-cache musl-dev binutils && rustup component add llvm-tools-preview
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY src src
COPY gen gen
RUN cargo run --release --manifest-path gen/Cargo.toml -- --seed 77 --events 300000 --clicks 75000 --output /tmp/pgo-data --threads 4 && \
    mkdir -p /tmp/pgo /tmp/pgo-out && \
    RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+crt-static -C relocation-model=static -C link-arg=-no-pie -C force-unwind-tables=no -C link-arg=-Wl,--gc-sections -C llvm-args=-x86-branches-within-32B-boundaries --cfg pgo_gen -C profile-generate=/tmp/pgo" \
    cargo build --release --locked --bin sessions && \
    SS_THREADS=8 /src/target/release/sessions /tmp/pgo-data /tmp/pgo-out && \
    "$(find "$(rustc --print sysroot)" -name "llvm-profdata" | head -n1)" merge -o /tmp/pgo/merged.profdata /tmp/pgo/*.profraw && \
    rm -rf /tmp/pgo-data /tmp/pgo-out /tmp/pgo/*.profraw
RUN RUSTFLAGS="-C target-cpu=cascadelake -C target-feature=+crt-static -C relocation-model=static -C link-arg=-no-pie -C force-unwind-tables=no -C link-arg=-Wl,--gc-sections -C llvm-args=-x86-branches-within-32B-boundaries -C llvm-args=-x86-pad-max-prefix-size=5 -C profile-use=/tmp/pgo/merged.profdata" \
    cargo build --release --locked --bin sessions && \
    strip --strip-all -R .eh_frame -R .eh_frame_hdr -R .gcc_except_table -R .comment -R .note.gnu.build-id /src/target/release/sessions

FROM scratch
COPY --from=build /src/target/release/sessions /sessions
USER 1000:1000
CMD ["/sessions"]
