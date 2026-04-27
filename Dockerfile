# Single build stage: clang (for BPF C codegen via libbpf-cargo) + rust.
FROM rust:bookworm AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        clang \
        pkg-config \
        autoconf \
        automake \
        autopoint \
        libtool \
        bison \
        flex \
        gawk \
        m4 \
    && rm -rf /var/lib/apt/lists/* && \
    rustup component add rustfmt

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY src/ src/
COPY benches/ benches/

RUN cargo build --release

# Minimal runtime.
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/profi /usr/local/bin/profi

EXPOSE 9401

ENTRYPOINT ["/usr/local/bin/profi"]
CMD ["--listen", "0.0.0.0:9401"]
