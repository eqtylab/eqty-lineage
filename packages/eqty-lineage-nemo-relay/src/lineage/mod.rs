//! Building an EQTY lineage graph, with no NeMo Relay in sight.
//!
//! This module deliberately knows nothing about Relay, ATOF, or coding agents. It is the layer
//! between `integrity`'s primitives and "a session's worth of lineage": you register assets, record
//! the computations that connected them, and get a signed manifest. Everything Relay-shaped lives
//! above it, in [`crate::classify`] and the recorder.
//!
//! Keeping that boundary is a bet that this layer wants to exist on its own eventually -- as a
//! feature of `integrity`, or as an `integrity-rs` beside `integrity-py`. Extraction from working
//! code is cheap; designing it speculatively is not. So it is written as though it were already
//! extracted, and nothing here should ever need a Relay type.
//!
//! # What `integrity` gives, and what it does not
//!
//! `integrity` has the primitives: statement constructors that compute their own CIDs, VC issuance,
//! `resolve_blobs`, and `generate_manifest`. What it does not have is the *composition* -- the fact
//! that "an asset" is three statements in a particular arrangement, and that the arrangement differs
//! depending on whether the asset has content. That composition is what `integrity-py` contributes
//! and what this module restates. Getting it wrong produces a manifest that parses, verifies, and
//! means something other than what happened.
//!
//! # One thing this does not copy from `integrity-py`
//!
//! The Python binding reaches its signer and blob store through process-global config (`with_cfg!`,
//! `active_signer`). That is workable for a script and wrong for us: this plugin is long-lived and
//! records many sessions at once, so a global active signer is a shared mutable that several
//! sessions would race over. It is also why `eqty_sdk.init()` is silently ignored on a second call,
//! and why comparing two recordings in Python needs two processes.
//!
//! Here the session owns its signer and its statements, and is passed explicitly.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use integrity::blob_store::{BlobStore, InMemoryStore};
use integrity::cid::blake3::blake3_cid_raw_binary;
use integrity::lineage::models::manifest::{Manifest, generate_manifest, resolve_blobs};
use integrity::lineage::models::statements::{
    ComputationStatement, DataStatement, EntityStatement, MetadataStatement, Statement,
    StatementTrait, VcStatement,
};
use integrity::signer::SignerType;
use integrity::vc;
use serde_json::Value;
use uuid::Uuid;

/// How many blobs `resolve_blobs` fetches at once. Ours are already in memory, so this only bounds
/// the futures it holds; the value matches what `integrity-py` passes.
const BLOB_CONCURRENCY: usize = 8;

/// A handle to something a computation can consume or produce.
///
/// For an asset with content this is its **content CID**, which means two identical files recorded
/// in different sessions are the same node in the merged graph. For an asset without content it is
/// the id of the statement that registered it, which is unique per registration. That asymmetry is
/// not ours -- it is what `eqty_sdk` does, and matching it is the point.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssetRef(String);

impl AssetRef {
    /// The underlying identifier, as it appears in a statement.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AssetRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One session's worth of lineage, accumulated in memory.
///
/// Statements are appended as they are made and the whole vector goes to `generate_manifest` at the
/// end. There is no database: `integrity-py` keeps statements in SQLite because it must serve
/// queries across many graphs, and the export ceiling people hit at roughly 10.9k statements is a
/// property of that retrieval, not of the manifest format. Recording one session needs none of it.
pub struct LineageSession {
    signer: SignerType,
    did: String,
    statements: Vec<Statement>,
    /// CID to bytes, for every blob a statement references. `generate_manifest` inlines these, so a
    /// metadata statement whose bytes are missing here becomes a CID nobody can resolve.
    blobs: HashMap<String, Vec<u8>>,
}

impl LineageSession {
    /// Start a session that signs with `signer`.
    pub fn new(signer: SignerType) -> Self {
        let did = signer.get_did_doc().id;
        Self {
            signer,
            did,
            statements: Vec::new(),
            blobs: HashMap::new(),
        }
    }

    /// The DID every statement in this session is registered by.
    pub fn did(&self) -> &str {
        &self.did
    }

    /// How many statements have been accumulated. For tests and coverage reporting.
    pub fn statement_count(&self) -> usize {
        self.statements.len()
    }

    /// Register an asset that has content, addressed by the hash of that content.
    ///
    /// The CID is computed over `content` exactly as given. Nothing is decoded, re-encoded, or
    /// wrapped in a JSON envelope first -- so this is correct for a file that is not valid UTF-8,
    /// which the Python path is not: it hands the asset constructor
    /// `content.decode("utf-8", errors="replace")`, and for binary content the manifest then asserts
    /// a `content-cid` that does not match the asset's own CID.
    pub async fn register_content(
        &mut self,
        content: &[u8],
        metadata: Value,
        at: Option<String>,
    ) -> Result<AssetRef> {
        let content_cid = blake3_cid_raw_binary(content)?;
        self.blobs.insert(content_cid.clone(), content.to_vec());

        let data = Statement::DataRegistration(
            DataStatement::create(vec![content_cid.clone()], self.did.clone(), at.clone()).await?,
        );
        self.push_with_proof(data, at.clone()).await?;

        // The metadata's subject is the *content CID*, not the statement that registered it. An
        // entity's metadata points at its statement instead; see `register_entity`.
        self.push_metadata(content_cid.clone(), metadata, at)
            .await?;

        Ok(AssetRef(content_cid))
    }

    /// Register an asset that has no content to hash -- an agent, a model, a tool.
    ///
    /// Identity is a fresh UUID, so two registrations of "the same" agent are two nodes. That is the
    /// existing behaviour and it is why anything that *can* be content-addressed should be.
    pub async fn register_entity(
        &mut self,
        metadata: Value,
        at: Option<String>,
    ) -> Result<AssetRef> {
        let uuid = Uuid::new_v4().to_string();
        let entity = Statement::EntityRegistration(
            EntityStatement::create(vec![uuid], self.did.clone(), at.clone()).await?,
        );
        let entity_id = entity.get_id();
        self.push_with_proof(entity, at.clone()).await?;

        // Unlike a content asset, an entity's metadata hangs off the registration statement.
        self.push_metadata(entity_id.clone(), metadata, at).await?;

        Ok(AssetRef(entity_id))
    }

    /// Record that some inputs produced some outputs.
    ///
    /// Empty outputs are rejected rather than recorded. An activity with no output is not evidence
    /// of anything -- it cannot be reached from any asset, so it is a node no reader can use, and it
    /// inflates the statement count while adding nothing a verifier can check.
    pub async fn record_computation(
        &mut self,
        inputs: &[AssetRef],
        outputs: &[AssetRef],
        at: Option<String>,
    ) -> Result<()> {
        if outputs.is_empty() {
            return Err(anyhow!("a computation with no outputs is not lineage"));
        }

        let computation = Statement::ComputationRegistration(
            ComputationStatement::create(
                None,
                inputs.iter().map(|input| input.0.clone()).collect(),
                outputs.iter().map(|output| output.0.clone()).collect(),
                self.did.clone(),
                None,
                self.did.clone(),
                at.clone(),
            )
            .await?,
        );
        self.push_with_proof(computation, at).await
    }

    /// Append a statement and the verifiable credential that proves who made it.
    ///
    /// The VC is what makes a manifest attributable rather than merely well-formed, and it is also
    /// why two recordings of identical work never produce identical statement CIDs: the credential
    /// carries a `validFrom` timestamp. Comparison has to go through the graph, never through bytes.
    async fn push_with_proof(&mut self, statement: Statement, at: Option<String>) -> Result<()> {
        let subject = statement.get_id();
        self.statements.push(statement);

        let credential = vc::issue_vc(&subject, self.signer.clone()).await?;
        let credential = serde_json::from_value(serde_json::to_value(credential)?)?;
        self.statements.push(Statement::CredentialRegistration(
            VcStatement::create(credential, self.did.clone(), at).await?,
        ));
        Ok(())
    }

    /// Append a metadata statement, and the bytes it commits to.
    ///
    /// `create_from_json` stores only a CID of the canonicalized metadata. Storing the matching
    /// bytes is not optional bookkeeping: without them the manifest carries a reference that
    /// resolves to nothing, and a reader cannot see what was claimed.
    async fn push_metadata(
        &mut self,
        subject: String,
        metadata: Value,
        at: Option<String>,
    ) -> Result<()> {
        let (metadata_cid, canonical) = integrity::cid::jcs::compute_jcs_cid(&metadata)?;
        self.blobs.insert(metadata_cid, canonical);

        self.statements.push(Statement::MetadataRegistration(
            MetadataStatement::create_from_json(subject, metadata, self.did.clone(), at).await?,
        ));
        Ok(())
    }

    /// Resolve every referenced blob and build the manifest.
    ///
    /// This is the same `generate_manifest` the Python SDK reaches through `Context.export()`, over
    /// the same `Statement` type, so the result verifies identically.
    pub async fn into_manifest(self) -> Result<Manifest> {
        // `InMemoryStore::put` is `unimplemented!()` upstream, which would panic if it were ever
        // called -- and this plugin runs in-process across a C ABI where a panic is undefined
        // behavior. It is never called: the map is populated directly as statements are made, and
        // `resolve_blobs` only reads. Constructing the struct rather than going through `put` is
        // what keeps that true.
        let store: Arc<dyn BlobStore + Send + Sync> = Arc::new(InMemoryStore { blobs: self.blobs });
        let blobs = resolve_blobs(&self.statements, store, BLOB_CONCURRENCY).await?;
        generate_manifest(true, self.statements, blobs).await
    }
}
