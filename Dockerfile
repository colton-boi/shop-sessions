FROM rust:1.100.0-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY src src
RUN cargo build --release --locked

FROM scratch
COPY --from=build /src/target/release/sessions /sessions
USER 1000:1000
CMD ["/sessions"]
