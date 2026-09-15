//! Building an EQTY lineage graph, with no NeMo Relay in sight.
//!
//! The layer between `integrity`'s primitives and "a session's worth of lineage": register assets,
//! record the computations that connected them, get a signed manifest. Everything Relay-shaped lives
//! above it, in [`crate::classify`] and the recorder. Nothing here should ever need a Relay type --
//! this layer is written as though it were already extracted into an `integrity-rs`.
//!
//! # What `integrity` gives, and what it does not
//!
//! `integrity` has the primitives: statement constructors that compute their own CIDs, VC issuance,
//! `resolve_blobs`, `generate_manifest`. What it lacks is the *composition* -- that an asset is
//! three statements in a particular arrangement, and that the arrangement differs depending on
//! whether the asset has content. Getting it wrong produces a manifest that parses, verifies, and
//! means something other than what happened.
//!
//! # One thing this does not copy from `integrity-py`
//!
//! The Python binding reaches its signer and blob store through process-global config (`with_cfg!`,
//! `active_signer`). This plugin is long-lived and records many sessions at once, so a global active
//! signer would be a shared mutable several sessions race over. Here the session owns its signer and
//! its statements, and is passed explicitly.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use integrity::blob_store::{BlobStore, InMemoryStore};
use integrity::cid::blake3::blake3_cid_raw_binary;
use integrity::cid::iroh::compute_iroh_collection_cid;
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
pub struct LineageSession {
    signer: SignerType,
    did: String,
    statements: Vec<Statement>,
    /// CID to bytes, for every blob a statement references. `generate_manifest` inlines these, so a
    /// metadata statement whose bytes are missing here becomes a CID nobody can resolve.
    blobs: HashMap<String, Vec<u8>>,
    /// Repeated content shares a registration to avoid duplicating statements and credentials.
    registered_content: HashSet<String>,
    /// `(subject, metadata CID)` pairs that already carry a `MetadataRegistration`.
    ///
    /// Deduped separately from the content, because the same bytes can be described differently --
    /// one file's contents seen at two paths is one node with two things said about it, and
    /// collapsing on content alone would silently drop the second path.
    described: HashSet<(String, String)>,
    /// Running total of the bytes in `blobs`.
    ///
    /// Tracked rather than summed on demand because it is read to decide whether a snapshot is worth
    /// taking, and walking every blob to answer "would walking every blob be expensive?" defeats the
    /// question.
    blob_bytes: usize,
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
            registered_content: HashSet::new(),
            described: HashSet::new(),
            blob_bytes: 0,
        }
    }

    /// Store one blob, keeping the byte total in step.
    ///
    /// Re-inserting the same CID is normal -- identical content is registered more than once in a
    /// session -- and the replaced entry's bytes must come back off the total or it drifts upward
    /// forever, which would throttle checkpoints for a session that never actually grew.
    fn put_blob(&mut self, cid: String, bytes: Vec<u8>) {
        self.blob_bytes += bytes.len();
        if let Some(replaced) = self.blobs.insert(cid, bytes) {
            self.blob_bytes -= replaced.len();
        }
    }

    /// Total bytes of blob content held.
    pub fn blob_bytes(&self) -> usize {
        self.blob_bytes
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
        self.put_blob(content_cid.clone(), content.to_vec());

        if self.registered_content.insert(content_cid.clone()) {
            let data = Statement::DataRegistration(
                DataStatement::create(vec![content_cid.clone()], self.did.clone(), at.clone())
                    .await?,
            );
            self.push_with_proof(data, at.clone()).await?;
        }

        // The metadata's subject is the *content CID*, not the statement that registered it. An
        // entity's metadata points at its statement instead; see `register_entity`.
        self.push_metadata(content_cid.clone(), metadata, at)
            .await?;

        Ok(AssetRef(content_cid))
    }

    /// Register an asset that *is* an iroh collection over content already in this session.
    ///
    /// Identity is the hashseq over the members' hashes under the collection multicodec, so it is
    /// the hash of no bytes stored here: `register_content` would address the hashseq blob as raw
    /// binary and the node would not be the collection it names.
    ///
    /// Both of the collection's own blobs are stored, and nothing needs to name them afterwards:
    /// `resolve_blobs` recognises the hashseq codec and pulls the meta blob and every member into
    /// the manifest from the CID on the `DataRegistration` alone.
    pub async fn register_collection(
        &mut self,
        members: &HashMap<String, String>,
        metadata: Value,
        at: Option<String>,
    ) -> Result<AssetRef> {
        let built = compute_iroh_collection_cid(members).await?;
        let collection_cid = built.collection.cid.clone();
        self.put_blob(collection_cid.clone(), built.collection.blob.to_vec());
        self.put_blob(built.meta.cid.clone(), built.meta.blob.to_vec());

        if self.registered_content.insert(collection_cid.clone()) {
            let data = Statement::DataRegistration(
                DataStatement::create(vec![collection_cid.clone()], self.did.clone(), at.clone())
                    .await?,
            );
            self.push_with_proof(data, at.clone()).await?;
        }

        self.push_metadata(collection_cid.clone(), metadata, at)
            .await?;

        Ok(AssetRef(collection_cid))
    }

    /// Store content without stating anything about it.
    ///
    /// For a payload that will be a member of a collection rather than a node of its own. The bytes
    /// have to be in the session for the collection to resolve, and a `DataRegistration` for each of
    /// them is the graph a collection exists to remove.
    pub fn stage_content(&mut self, content: &[u8]) -> Result<AssetRef> {
        let content_cid = blake3_cid_raw_binary(content)?;
        self.put_blob(content_cid.clone(), content.to_vec());
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

    /// Record that some inputs produced some outputs, and say something about the activity itself.
    ///
    /// The metadata's subject is the computation statement, not an asset. That is the difference
    /// between "this activity was performed by the researcher subagent" and "this activity consumed
    /// the researcher subagent" -- the second is what putting an agent in `inputs` would assert, and
    /// it is false. PROV keeps association and usage apart, and so does this.
    pub async fn record_computation_described(
        &mut self,
        inputs: &[AssetRef],
        outputs: &[AssetRef],
        describes: Value,
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
        let subject = computation.get_id();
        self.push_with_proof(computation, at.clone()).await?;
        self.push_metadata(subject, describes, at).await
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

    /// Keep canonical metadata bytes so their CID resolves in the manifest.
    /// Sign the metadata statement so its claims are attributable to the session's signer.
    async fn push_metadata(
        &mut self,
        subject: String,
        mut metadata: Value,
        at: Option<String>,
    ) -> Result<()> {
        encode_nested_values(&mut metadata);
        let (metadata_cid, canonical) = integrity::cid::jcs::compute_jcs_cid(&metadata)?;
        self.put_blob(metadata_cid.clone(), canonical);

        // Same subject, same claim, already stated. Saying it twice adds nothing a reader can use.
        if !self.described.insert((subject.clone(), metadata_cid)) {
            return Ok(());
        }

        let statement = Statement::MetadataRegistration(
            MetadataStatement::create_from_json(subject, metadata, self.did.clone(), at.clone())
                .await?,
        );
        self.push_with_proof(statement, at).await
    }

    /// Snapshot without consuming the session; existing statements retain their credentials.
    /// Copies all statements and blobs, so callers must pace snapshots by recording size.
    pub async fn snapshot(&self) -> Result<Manifest> {
        Self::build(self.statements.clone(), self.blobs.clone()).await
    }

    /// The same manifest, consuming the session. For an export, which has no later use for it.
    pub async fn into_manifest(self) -> Result<Manifest> {
        Self::build(self.statements, self.blobs).await
    }

    /// Resolve every referenced blob and build the manifest.
    async fn build(
        statements: Vec<Statement>,
        blobs: HashMap<String, Vec<u8>>,
    ) -> Result<Manifest> {
        // `InMemoryStore::put` is `unimplemented!()` upstream, which would panic if it were ever
        // called -- and this plugin runs in-process across a C ABI where a panic is undefined
        // behavior. It is never called: the map is populated directly as statements are made, and
        // `resolve_blobs` only reads. Constructing the struct rather than going through `put` is
        // what keeps that true.
        let store: Arc<dyn BlobStore + Send + Sync> = Arc::new(InMemoryStore { blobs });
        let resolved = resolve_blobs(&statements, store, BLOB_CONCURRENCY).await?;
        generate_manifest(true, statements, resolved).await
    }
}

/// JSON-encode every nested metadata value, because the graph explorer displays each value as a
/// string and renders an object or an array as the literal text `[object Object]`.
///
/// Applied here rather than at each call site: this is the one place every metadata statement passes
/// through, and doing it per site is how the same bug shipped three times.
///
/// Only the top level is rewritten. A nested object becomes a JSON string that still contains its
/// own structure, so nothing is lost. The node's *content* stays real JSON either way, which keeps
/// its identity comparable across runs. `null` is left alone -- `withheldBecause: null` is a value a
/// reader acts on, and it renders as itself.
///
/// Same treatment `eqty-lineage-langchain` applies before values reach the SDK.
fn encode_nested_values(metadata: &mut Value) {
    let Some(fields) = metadata.as_object_mut() else {
        return;
    };
    for value in fields.values_mut() {
        if value.is_object() || value.is_array() {
            // A value that will not serialize cannot be rendered either, so dropping to `null` is
            // the honest outcome -- and `serde_json` only fails here on a map with non-string keys,
            // which JSON cannot express and this crate never builds.
            *value = match serde_json::to_string(value) {
                Ok(text) => Value::String(text),
                Err(_) => Value::Null,
            };
        }
    }
}
