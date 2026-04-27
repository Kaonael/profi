// SPDX-License-Identifier: Apache-2.0

//! PEM loading helpers shared by the OTLP client exporter and the
//! HTTPS/mTLS `/metrics` server. Keeps `fs::read` + `rustls-pemfile` decoding
//! in one place so both code paths behave identically (and fail identically).

use std::fs;
use std::io::BufReader;
use std::path::Path;

use anyhow::{anyhow, Context, Result};

/// Read the raw PEM bytes of a file. Returned as `Vec<u8>` because tonic's
/// `Certificate::from_pem` / `Identity::from_pem` take bytes, not parsed certs.
pub fn read_pem_file(path: impl AsRef<Path>) -> Result<Vec<u8>> {
    let path = path.as_ref();
    fs::read(path).with_context(|| format!("read PEM file {}", path.display()))
}

/// Parse a PEM certificate chain into rustls `CertificateDer`s.
pub fn load_cert_chain(
    path: impl AsRef<Path>,
) -> Result<Vec<rustls::pki_types::CertificateDer<'static>>> {
    let path = path.as_ref();
    let file = fs::File::open(path)
        .with_context(|| format!("open certificate chain {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let certs: Vec<_> = rustls_pemfile::certs(&mut reader)
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("parse certificate chain {}", path.display()))?;
    if certs.is_empty() {
        return Err(anyhow!(
            "no certificates found in {} (empty or malformed PEM)",
            path.display()
        ));
    }
    Ok(certs)
}

/// Parse a PEM private key (PKCS#1, PKCS#8, or SEC1) into a rustls
/// `PrivateKeyDer`. Returns the first key found.
pub fn load_private_key(
    path: impl AsRef<Path>,
) -> Result<rustls::pki_types::PrivateKeyDer<'static>> {
    let path = path.as_ref();
    let file =
        fs::File::open(path).with_context(|| format!("open private key {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let key = rustls_pemfile::private_key(&mut reader)
        .with_context(|| format!("parse private key {}", path.display()))?
        .ok_or_else(|| anyhow!("no private key found in {}", path.display()))?;
    Ok(key)
}
