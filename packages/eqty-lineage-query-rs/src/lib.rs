//! Native accelerator for `eqty_lineage.query`.
//!
//! A straight port of `eqty_lineage/query/engine.py` -- the same semi-naive fixpoint, the same
//! semirings, the same flow orientation. Deliberately not a different engine: the Python
//! implementation is the reference oracle, and a port that changed the algorithm could not be checked
//! against it.
//!
//! The boundary is whole queries, not individual rules. The fixpoint is essentially all of the cost,
//! so crossing PyO3 once per query keeps marshalling irrelevant; exposing `evaluate(rules, edb)` would
//! instead marshal set-of-set annotations across the boundary on every iteration.
//!
//! Two deliberate divergences from the Python reference, both observable:
//!   * `counting` saturates at `u64::MAX` where Python uses arbitrary-precision integers. No real
//!     session has come close, but a pathological graph would differ.
//!   * strings are interned to `u32` internally; identity is by value either way.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::collections::HashMap;

// ---------------------------------------------------------------- interning
#[derive(Default)]
struct Interner {
    map: HashMap<String, u32>,
    names: Vec<String>,
}

impl Interner {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(s.to_string());
        self.map.insert(s.to_string(), id);
        id
    }
    fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }
}

// ---------------------------------------------------------------- semirings
trait Semiring: Clone + PartialEq {
    fn zero() -> Self;
    fn one() -> Self;
    fn plus(&self, other: &Self) -> Self;
    fn times(&self, other: &Self) -> Self;
    fn lift(fact: u32) -> Self;
}

#[derive(Clone, PartialEq)]
struct Boolean(bool);
impl Semiring for Boolean {
    fn zero() -> Self { Boolean(false) }
    fn one() -> Self { Boolean(true) }
    fn plus(&self, o: &Self) -> Self { Boolean(self.0 || o.0) }
    fn times(&self, o: &Self) -> Self { Boolean(self.0 && o.0) }
    fn lift(_f: u32) -> Self { Boolean(true) }
}

#[derive(Clone, PartialEq)]
struct Counting(u64);
impl Semiring for Counting {
    fn zero() -> Self { Counting(0) }
    fn one() -> Self { Counting(1) }
    fn plus(&self, o: &Self) -> Self { Counting(self.0.saturating_add(o.0)) }
    fn times(&self, o: &Self) -> Self { Counting(self.0.saturating_mul(o.0)) }
    fn lift(_f: u32) -> Self { Counting(1) }
}

/// Subset-minimal monomials, each a sorted set of fact ids. The outer vector is kept sorted so
/// `PartialEq` is set equality -- which is what the fixpoint's termination test needs.
#[derive(Clone, PartialEq)]
struct Absorptive(Vec<Vec<u32>>);

fn is_subset(a: &[u32], b: &[u32]) -> bool {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] { i += 1; j += 1; }
        else if a[i] > b[j] { j += 1; }
        else { return false; }
    }
    i == a.len()
}

fn union_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] { out.push(a[i]); i += 1; }
        else if a[i] > b[j] { out.push(b[j]); j += 1; }
        else { out.push(a[i]); i += 1; j += 1; }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Cap on retained minimal witnesses, 0 meaning unbounded.
///
/// The witness *basis* can be exponentially large -- one real session has a pair with 118,096 minimal
/// witnesses in a 2,465-pair graph -- and this is not an artefact of the representation: computing the
/// minimal elements of a set family provably requires exponential size even as a ZDD, independent of
/// variable ordering. No faster code avoids an output that large.
///
/// Capping changes what is computed: the result is *k genuine minimal witnesses*, not the complete
/// basis. That breaks the semiring laws (`plus` stops being associative once truncation kicks in), so
/// it is an approximation with a stated bound, not an optimisation. Smallest monomials are kept first,
/// which retains the most general explanations -- the ones a reader wants.
// Must match `eqty_lineage.query.semiring.WITNESS_CAP`. A different default here would make the two
// backends disagree on exactly the sessions where the cap matters, which is the hardest case to
// notice.
static WITNESS_CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(64);

fn witness_cap() -> usize {
    WITNESS_CAP.load(std::sync::atomic::Ordering::Relaxed)
}

fn minimalize(mut ms: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    ms.sort_by_key(|m| m.len());
    let mut out: Vec<Vec<u32>> = Vec::new();
    'outer: for cand in ms {
        for kept in &out {
            if is_subset(kept, &cand) {
                continue 'outer;
            }
        }
        out.push(cand);
    }
    out.sort();
    out.dedup();
    let cap = witness_cap();
    if cap > 0 && out.len() > cap {
        // keep the shortest: fewest facts required is the most general witness
        out.sort_by_key(|m| m.len());
        out.truncate(cap);
        out.sort();
    }
    out
}

impl Semiring for Absorptive {
    fn zero() -> Self { Absorptive(Vec::new()) }
    fn one() -> Self { Absorptive(vec![Vec::new()]) }
    fn plus(&self, o: &Self) -> Self {
        let mut all = self.0.clone();
        all.extend(o.0.iter().cloned());
        Absorptive(minimalize(all))
    }
    fn times(&self, o: &Self) -> Self {
        let mut all = Vec::with_capacity(self.0.len() * o.0.len());
        for x in &self.0 {
            for y in &o.0 {
                all.push(union_sorted(x, y));
            }
        }
        Absorptive(minimalize(all))
    }
    fn lift(f: u32) -> Self { Absorptive(vec![vec![f]]) }
}

/// Determination: the set of resolutions under which a fact holds, as a bitmask.
///
/// `plus` unions (alternative derivations each contribute their runs), `times` intersects (a path needs
/// every edge present in the *same* run). Both are single machine instructions, which makes this the
/// cheapest semiring here -- cheaper than boolean, since the carrier is a word either way but no
/// branching is involved.
///
/// `one` is the full run set, not a singleton: a fact present in every run is the multiplicative
/// identity because intersecting with it changes nothing. Sixty-four runs is the ceiling for this
/// representation; beyond that the carrier needs to become a `Vec<u64>`.
#[derive(Clone, PartialEq)]
struct Determination(u64);
impl Semiring for Determination {
    fn zero() -> Self { Determination(0) }
    fn one() -> Self { Determination(u64::MAX) }
    fn plus(&self, o: &Self) -> Self { Determination(self.0 | o.0) }
    fn times(&self, o: &Self) -> Self { Determination(self.0 & o.0) }
    fn lift(_f: u32) -> Self { Determination(0) }
}

/// Integrity lattice: `plus` and `times` are both `max`. One tainted derivation taints the result and
/// no alternative can launder it -- the dual of confidentiality, where `plus` is `min`.
#[derive(Clone, PartialEq)]
struct Integrity(u8);
impl Semiring for Integrity {
    fn zero() -> Self { Integrity(0) }
    fn one() -> Self { Integrity(0) }
    fn plus(&self, o: &Self) -> Self { Integrity(self.0.max(o.0)) }
    fn times(&self, o: &Self) -> Self { Integrity(self.0.max(o.0)) }
    fn lift(_f: u32) -> Self { Integrity(0) }
}

// ---------------------------------------------------------------- evaluation
type Edges<S> = HashMap<(u32, u32), S>;

/// Semi-naive transitive closure of the flow relation.
///
/// Specialized to `influenced(X,Y) :- edge(X,Y).` / `influenced(X,Z) :- influenced(X,Y), edge(Y,Z).`
/// The general rule machinery in the Python engine exists so callers can write their own rules; the
/// closure is the only shape whose cost justifies crossing into native code.
fn closure<S: Semiring>(edges: &Edges<S>, max_iterations: usize) -> Option<HashMap<(u32, u32), S>> {
    // index edges by source for the join
    let mut by_source: HashMap<u32, Vec<(u32, S)>> = HashMap::new();
    for (&(a, b), ann) in edges.iter() {
        by_source.entry(a).or_default().push((b, ann.clone()));
    }

    let mut known: HashMap<(u32, u32), S> = edges.clone();
    let mut delta: HashMap<(u32, u32), S> = edges.clone();

    for _ in 0..max_iterations {
        let mut pending: HashMap<(u32, u32), S> = HashMap::new();

        for (&(x, y), ann) in delta.iter() {
            if let Some(nexts) = by_source.get(&y) {
                for (z, edge_ann) in nexts {
                    let combined = ann.times(edge_ann);
                    match pending.get_mut(&(x, *z)) {
                        Some(slot) => *slot = slot.plus(&combined),
                        None => { pending.insert((x, *z), combined); }
                    }
                }
            }
        }

        let mut changed: HashMap<(u32, u32), S> = HashMap::new();
        for (key, ann) in pending {
            let merged = match known.get(&key) {
                Some(prev) => {
                    let m = prev.plus(&ann);
                    // Termination is by *annotation* equality, not tuple presence: a tuple can be
                    // rederived with a larger annotation, and stopping at first sighting would
                    // silently truncate its provenance.
                    if m == *prev { continue; }
                    m
                }
                None => ann,
            };
            known.insert(key, merged.clone());
            changed.insert(key, merged);
        }

        if changed.is_empty() {
            return Some(known);
        }
        delta = changed;
    }
    None
}

// ---------------------------------------------------------------- provenance circuits
/// A reference inside a derivation: either a base fact, or another derived tuple.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Ref {
    Fact(u32),
    Tuple(u32, u32),
}

/// The derivations of every tuple, shared rather than expanded.
///
/// The expanded polynomial for a pair can run to hundreds of millions of monomials; the circuit that
/// generates it is polynomial, because each derivation step contributes exactly one product term.
///
/// Built by *recording* an ordinary semi-naive evaluation rather than by running a circuit-valued
/// fixpoint. Circuit nodes are syntactically distinct even when semantically equal, so a fixpoint over
/// them would never satisfy its termination test. Semi-naive also enumerates each body combination
/// exactly once across the whole run -- the delta partition guarantees it -- so recording as we go
/// yields every derivation with no extra pass.
///
/// Acyclicity comes free: an instantiation generated at round *k* references only tuples known before
/// round *k*, so the tuple graph is stratified by round even when the underlying edge graph is not.
struct Circuit {
    derivations: HashMap<(u32, u32), Vec<Vec<Ref>>>,
}

impl Circuit {
    fn evaluate<S: Semiring>(&self) -> HashMap<(u32, u32), S> {
        let mut memo: HashMap<(u32, u32), S> = HashMap::new();
        let keys: Vec<(u32, u32)> = self.derivations.keys().copied().collect();
        for key in keys {
            let mut stack = std::collections::HashSet::new();
            self.value_of::<S>(key, &mut memo, &mut stack);
        }
        memo
    }

    fn value_of<S: Semiring>(
        &self,
        key: (u32, u32),
        memo: &mut HashMap<(u32, u32), S>,
        stack: &mut std::collections::HashSet<(u32, u32)>,
    ) -> S {
        if let Some(v) = memo.get(&key) {
            return v.clone();
        }
        // HashSet, not Vec: a linear membership scan here is O(depth) per call and turns evaluation
        // quadratic on the long derivation chains real lineage produces.
        if stack.contains(&key) {
            // Defensive: the round-stratified construction should make this unreachable. Returning zero
            // rather than recursing keeps a malformed circuit from blowing the stack.
            return S::zero();
        }
        stack.insert(key);
        let mut total = S::zero();
        if let Some(instantiations) = self.derivations.get(&key) {
            for body in instantiations {
                let mut product = S::one();
                for r in body {
                    let v = match *r {
                        Ref::Fact(f) => S::lift(f),
                        Ref::Tuple(a, b) => self.value_of::<S>((a, b), memo, stack),
                    };
                    product = product.times(&v);
                }
                total = total.plus(&product);
            }
        }
        stack.remove(&key);
        memo.insert(key, total.clone());
        total
    }
}

/// Semi-naive closure that records its derivations instead of annotating them.
///
/// Runs in the boolean semiring -- the cheapest available -- because the only thing the loop needs to
/// decide is whether a tuple is new. Provenance is recovered afterwards by evaluating the circuit, at
/// any semiring, as many times as you like from one traversal.
fn build_circuit(edges: &HashMap<(u32, u32), u32>, max_iterations: usize) -> Option<Circuit> {
    let mut by_source: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    for (&(a, b), &fact) in edges.iter() {
        by_source.entry(a).or_default().push((b, fact));
    }

    let mut derivations: HashMap<(u32, u32), Vec<Vec<Ref>>> = HashMap::new();
    let mut known: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut delta: Vec<(u32, u32)> = Vec::new();

    for (&key, &fact) in edges.iter() {
        derivations.entry(key).or_default().push(vec![Ref::Fact(fact)]);
        if known.insert(key) {
            delta.push(key);
        }
    }

    for _ in 0..max_iterations {
        let mut next: Vec<(u32, u32)> = Vec::new();
        for &(x, y) in &delta {
            if let Some(nexts) = by_source.get(&y) {
                for &(z, fact) in nexts {
                    derivations
                        .entry((x, z))
                        .or_default()
                        .push(vec![Ref::Tuple(x, y), Ref::Fact(fact)]);
                    if known.insert((x, z)) {
                        next.push((x, z));
                    }
                }
            }
        }
        if next.is_empty() {
            return Some(Circuit { derivations });
        }
        delta = next;
    }
    None
}

// ---------------------------------------------------------------- input
/// Mirrors `FLOW_ORIENTATION`. PROV predicates point backwards in time, so the flow relation reverses
/// them. `wasInvalidatedBy` is the exact inverse of `wasDerivedFrom` and is dropped: keeping both puts
/// a 2-cycle in the flow relation for every twice-edited file.
fn orientation(predicate: &str) -> Option<bool> {
    match predicate {
        "prov:wasInvalidatedBy" | "eqty:hasPath" | "eqty:assetType" | "eqty:label" => None,
        "eqty:triggered" => Some(false),
        _ => Some(true),
    }
}

/// Last 8 characters, matching Python's `t.subject[-8:]`. Character-wise, not byte-wise: CIDs are ASCII
/// but a slice into the middle of a multi-byte character would panic.
fn tail8(s: &str) -> String {
    let n = s.chars().count();
    s.chars().skip(n.saturating_sub(8)).collect()
}

fn build<S: Semiring>(triples: &[(String, String, String)], interner: &mut Interner) -> Edges<S> {
    let mut edges: Edges<S> = HashMap::new();
    for (s, p, o) in triples {
        let reverse = match orientation(p) { Some(r) => r, None => continue };
        let (a, b) = if reverse { (o.as_str(), s.as_str()) } else { (s.as_str(), o.as_str()) };
        // Fact identifiers must match the Python reference exactly -- it abbreviates endpoints to the
        // last 8 characters for readability, and witness sets are compared by these names.
        let fact = interner.intern(&format!("{}:{}->{}", p, tail8(s), tail8(o)));
        edges.insert((interner.intern(a), interner.intern(b)), S::lift(fact));
    }
    edges
}

fn err(msg: &str) -> PyErr {
    PyRuntimeError::new_err(msg.to_string())
}

const NO_FIXPOINT: &str = "no fixpoint within max_iterations (non-absorptive semiring over a cycle?)";

// ---------------------------------------------------------------- python API
/// Reachable pairs. `[(subject, object)]`.
#[pyfunction]
#[pyo3(signature = (triples, max_iterations = 1000))]
fn closure_bool(
    triples: Vec<(String, String, String)>,
    max_iterations: usize,
) -> PyResult<Vec<(String, String)>> {
    let mut interner = Interner::default();
    let edges = build::<Boolean>(&triples, &mut interner);
    let derived = closure(&edges, max_iterations).ok_or_else(|| err(NO_FIXPOINT))?;
    Ok(derived
        .keys()
        .map(|&(a, b)| (interner.name(a).to_string(), interner.name(b).to_string()))
        .collect())
}

/// Reachable pairs with their derivation counts. `[(subject, object, count)]`.
#[pyfunction]
#[pyo3(signature = (triples, max_iterations = 1000))]
fn closure_count(
    triples: Vec<(String, String, String)>,
    max_iterations: usize,
) -> PyResult<Vec<(String, String, u64)>> {
    let mut interner = Interner::default();
    let edges = build::<Counting>(&triples, &mut interner);
    let derived = closure(&edges, max_iterations).ok_or_else(|| err(NO_FIXPOINT))?;
    Ok(derived
        .iter()
        .map(|(&(a, b), c)| (interner.name(a).to_string(), interner.name(b).to_string(), c.0))
        .collect())
}

/// Reachable pairs with the *number* of minimal witness sets. Sizes rather than the sets themselves:
/// a large session has tens of thousands of pairs, and callers that want the witnesses want them for
/// one pair, not all of them -- see [`witnesses`].
#[pyfunction]
#[pyo3(signature = (triples, max_iterations = 1000))]
fn closure_absorptive_sizes(
    triples: Vec<(String, String, String)>,
    max_iterations: usize,
) -> PyResult<Vec<(String, String, usize)>> {
    let mut interner = Interner::default();
    let edges = build::<Absorptive>(&triples, &mut interner);
    let derived = closure(&edges, max_iterations).ok_or_else(|| err(NO_FIXPOINT))?;
    Ok(derived
        .iter()
        .map(|(&(a, b), w)| (interner.name(a).to_string(), interner.name(b).to_string(), w.0.len()))
        .collect())
}

/// The minimal witness sets for one pair, as lists of fact identifiers.
#[pyfunction]
#[pyo3(signature = (triples, source, target, max_iterations = 1000))]
fn witnesses(
    triples: Vec<(String, String, String)>,
    source: String,
    target: String,
    max_iterations: usize,
) -> PyResult<Vec<Vec<String>>> {
    let mut interner = Interner::default();
    let edges = build::<Absorptive>(&triples, &mut interner);
    let derived = closure(&edges, max_iterations).ok_or_else(|| err(NO_FIXPOINT))?;

    let (s, t) = match (interner.map.get(&source), interner.map.get(&target)) {
        (Some(&s), Some(&t)) => (s, t),
        _ => return Ok(Vec::new()),
    };
    Ok(derived
        .get(&(s, t))
        .map(|w| {
            w.0.iter()
                .map(|m| m.iter().map(|&f| interner.name(f).to_string()).collect())
                .collect()
        })
        .unwrap_or_default())
}

/// Pairs whose target is reachable from an untrusted source, under the integrity semiring.
#[pyfunction]
#[pyo3(signature = (triples, untrusted, max_iterations = 1000))]
fn taint(
    triples: Vec<(String, String, String)>,
    untrusted: Vec<String>,
    max_iterations: usize,
) -> PyResult<Vec<(String, String)>> {
    let mut interner = Interner::default();
    let marked: std::collections::HashSet<&str> = untrusted.iter().map(|s| s.as_str()).collect();

    let mut edges: Edges<Integrity> = HashMap::new();
    for (s, p, o) in &triples {
        let reverse = match orientation(p) { Some(r) => r, None => continue };
        let (a, b) = if reverse { (o.as_str(), s.as_str()) } else { (s.as_str(), o.as_str()) };
        let level = if marked.contains(a) { 2u8 } else { 0u8 };
        edges.insert((interner.intern(a), interner.intern(b)), Integrity(level));
    }

    let derived = closure(&edges, max_iterations).ok_or_else(|| err(NO_FIXPOINT))?;
    Ok(derived
        .iter()
        .filter(|(_, level)| level.0 >= 2)
        .map(|(&(a, b), _)| (interner.name(a).to_string(), interner.name(b).to_string()))
        .collect())
}

/// For each reachable pair, the set of runs in which that influence holds.
///
/// Edge masks are supplied per triple rather than derived from a fact identifier: run membership is a
/// property of the *observation*, not of the edge's content, so it cannot be recovered from the triple
/// itself.
#[pyfunction]
#[pyo3(signature = (triples, masks, n_runs, max_iterations = 1000))]
fn closure_determination(
    triples: Vec<(String, String, String)>,
    masks: Vec<u64>,
    n_runs: usize,
    max_iterations: usize,
) -> PyResult<Vec<(String, String, u64)>> {
    if masks.len() != triples.len() {
        return Err(err("masks must be the same length as triples"));
    }
    if n_runs > 64 {
        return Err(err("more than 64 runs needs a wider carrier than u64"));
    }
    let full = if n_runs == 64 { u64::MAX } else { (1u64 << n_runs) - 1 };

    let mut interner = Interner::default();
    let mut edges: Edges<Determination> = HashMap::new();
    for ((s, p, o), mask) in triples.iter().zip(masks.iter()) {
        let reverse = match orientation(p) { Some(r) => r, None => continue };
        let (a, b) = if reverse { (o.as_str(), s.as_str()) } else { (s.as_str(), o.as_str()) };
        let key = (interner.intern(a), interner.intern(b));
        let slot = edges.entry(key).or_insert(Determination(0));
        *slot = Determination(slot.0 | (mask & full));
    }

    let derived = closure(&edges, max_iterations).ok_or_else(|| err(NO_FIXPOINT))?;
    Ok(derived
        .iter()
        .filter(|(_, m)| m.0 & full != 0)
        .map(|(&(a, b), m)| (interner.name(a).to_string(), interner.name(b).to_string(), m.0 & full))
        .collect())
}

/// Set the retained-witness cap; 0 restores exact (unbounded) evaluation. Returns the previous value.
#[pyfunction]
fn set_witness_cap(cap: usize) -> usize {
    WITNESS_CAP.swap(cap, std::sync::atomic::Ordering::Relaxed)
}

/// The alternatives report, computed from a single circuit traversal.
///
/// The direct path runs the closure twice, once per semiring. Here the fixpoint runs once, in boolean,
/// and both semirings are recovered by evaluating the recorded circuit -- which is also what keeps the
/// absorptive pass from re-minimalising the same intermediate products at every step.
#[pyfunction]
#[pyo3(signature = (triples, max_iterations = 1000))]
fn measure_circuit(
    triples: Vec<(String, String, String)>,
    max_iterations: usize,
) -> PyResult<(usize, usize, u64, u64, usize)> {
    let mut interner = Interner::default();
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for (s, p, o) in &triples {
        let reverse = match orientation(p) { Some(r) => r, None => continue };
        let (a, b) = if reverse { (o.as_str(), s.as_str()) } else { (s.as_str(), o.as_str()) };
        let fact = interner.intern(&format!("{}:{}->{}", p, tail8(s), tail8(o)));
        edges.insert((interner.intern(a), interner.intern(b)), fact);
    }

    let circuit = build_circuit(&edges, max_iterations).ok_or_else(|| err(NO_FIXPOINT))?;
    let counts = circuit.evaluate::<Counting>();
    let wit = circuit.evaluate::<Absorptive>();

    let multi = counts.values().filter(|c| c.0 > 1).count();
    let max_derivations = counts.values().map(|c| c.0).max().unwrap_or(0);
    let monomials_total: u64 = wit.values().map(|w| w.0.len() as u64).sum();
    let max_monomials = wit.values().map(|w| w.0.len()).max().unwrap_or(0);
    Ok((counts.len(), multi, max_derivations, monomials_total, max_monomials))
}

/// `(reachable_pairs, pairs_multi_derivation, max_derivations, monomials_total, max_monomials)`.
///
/// Computed natively so the measurement never marshals the closure -- the whole point of the sweep is
/// to run it over many sessions.
#[pyfunction]
#[pyo3(signature = (triples, max_iterations = 1000))]
fn measure(
    triples: Vec<(String, String, String)>,
    max_iterations: usize,
) -> PyResult<(usize, usize, u64, u64, usize)> {
    // Counting is non-idempotent, and semi-naive double counts re-derivations for such semirings, so
    // both are evaluated over a recorded circuit instead. See `measure_circuit`.
    return measure_circuit(triples, max_iterations);
    #[allow(unreachable_code)]
    let (counts, wit) = (HashMap::<(u32,u32), Counting>::new(), HashMap::<(u32,u32), Absorptive>::new());

    let multi = counts.values().filter(|c| c.0 > 1).count();
    let max_derivations = counts.values().map(|c| c.0).max().unwrap_or(0);
    let monomials_total: u64 = wit.values().map(|w| w.0.len() as u64).sum();
    let max_monomials = wit.values().map(|w| w.0.len()).max().unwrap_or(0);

    Ok((counts.len(), multi, max_derivations, monomials_total, max_monomials))
}

#[pymodule]
fn eqty_lineage_query_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(closure_bool, m)?)?;
    m.add_function(wrap_pyfunction!(closure_count, m)?)?;
    m.add_function(wrap_pyfunction!(closure_absorptive_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(witnesses, m)?)?;
    m.add_function(wrap_pyfunction!(taint, m)?)?;
    m.add_function(wrap_pyfunction!(closure_determination, m)?)?;
    m.add_function(wrap_pyfunction!(measure_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(set_witness_cap, m)?)?;
    m.add_function(wrap_pyfunction!(measure, m)?)?;
    m.add("__doc__", "Native accelerator for eqty_lineage.query.")?;
    Ok(())
}
