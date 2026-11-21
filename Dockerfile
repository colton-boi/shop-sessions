FROM rust:1.100.0-slim-bookworm AS build
WORKDIR /gen
COPY gen ./
RUN RUSTFLAGS="-C target-feature=+crt-static" cargo build --release --locked --target x86_64-unknown-linux-gnu && cp target/x86_64-unknown-linux-gnu/release/generate /generate

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY src src
RUN cargo build --release --locked && cp /generate target/release/

FROM scratch
COPY --from=build /src/target/release/sessions /sessions
COPY --from=build /src/target/release/probe /probe
COPY --from=build /src/target/release/generate /generate
USER 1000:1000
CMD ["/probe"]
