//! The request types, their pre-serialized bodies, and the seeded request
//! sequence.

use {
    crate::{
        history::History,
        world::{TxKind, World, Zipf, index, range},
    },
    base64::{Engine, prelude::BASE64_STANDARD},
    hyper::body::Bytes,
    rand_chacha::{ChaCha8Rng, rand_core::RngCore},
    serde_json::{Value, json},
    solana_hash::Hash,
    solana_program_pack::Pack,
    solana_pubkey::Pubkey,
    solana_signature::Signature,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Light,
    Medium,
    Heavy,
    Exec,
}

pub const CLASSES: [Class; 4] = [Class::Light, Class::Medium, Class::Heavy, Class::Exec];

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Light => "light",
            Class::Medium => "medium",
            Class::Heavy => "heavy",
            Class::Exec => "exec",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Mixed,
    Class(Class),
}

impl Mode {
    pub fn default_requests(self) -> usize {
        match self {
            Mode::Mixed => 400_000,
            Mode::Class(Class::Light) => 1_000_000,
            Mode::Class(Class::Medium) => 60_000,
            Mode::Class(Class::Heavy) => 3_000,
            Mode::Class(Class::Exec) => 60_000,
        }
    }

    /// Share of each class in this mode, in parts per thousand.
    fn class_share(self, class: Class) -> u64 {
        match self {
            Mode::Mixed => match class {
                Class::Light => 865,
                Class::Medium => 70,
                Class::Heavy => 5,
                Class::Exec => 60,
            },
            Mode::Class(only) if only == class => 1000,
            Mode::Class(_) => 0,
        }
    }
}

pub struct RequestType {
    pub name: &'static str,
    pub class: Class,
    /// Weight within the class, in percent.
    pub weight: u64,
}

const fn request_type(name: &'static str, class: Class, weight: u64) -> RequestType {
    RequestType {
        name,
        class,
        weight,
    }
}

pub const REQUEST_TYPES: [RequestType; 20] = [
    request_type("getSlot", Class::Light, 20),
    request_type("getVersion", Class::Light, 5),
    request_type("getLatestBlockhash", Class::Light, 20),
    request_type("getBalance", Class::Light, 25),
    request_type("getAccountInfo", Class::Light, 30),
    request_type("getMultipleAccounts", Class::Medium, 30),
    request_type("getSignatureStatuses", Class::Medium, 25),
    request_type("getTransaction", Class::Medium, 20),
    request_type("getTokenAccountsByOwner", Class::Medium, 15),
    request_type("getSignaturesForAddress", Class::Medium, 10),
    request_type("getProgramAccounts.scan", Class::Heavy, 20),
    request_type("getProgramAccounts.mint", Class::Heavy, 15),
    request_type("getTokenAccountsByOwner.whale", Class::Heavy, 15),
    request_type("getBlock.full", Class::Heavy, 20),
    request_type("getBlock.signatures", Class::Heavy, 15),
    request_type("getSignaturesForAddress.token", Class::Heavy, 15),
    request_type("simulate.batch", Class::Exec, 40),
    request_type("simulate.cpi", Class::Exec, 25),
    request_type("simulate.swap", Class::Exec, 25),
    request_type("simulate.fail", Class::Exec, 10),
];

/// Token-2022 transfers in the batch and fail simulation types.
pub const BATCH_TRANSFERS: usize = 40;
/// The getProgramAccounts by mint queries use mints around this popularity
/// rank, which have about 3,000 holders in the full world.
const MINT_QUERY_RANK: usize = 14;

pub fn body(method: &str, params: Value) -> Bytes {
    Bytes::from(json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).to_string())
}

pub fn get_balance(address: &Pubkey) -> Bytes {
    body("getBalance", json!([address.to_string()]))
}

pub fn get_account_info(address: &Pubkey) -> Bytes {
    body(
        "getAccountInfo",
        json!([address.to_string(), {"encoding": "base64"}]),
    )
}

pub fn get_multiple_accounts(addresses: &[Pubkey]) -> Bytes {
    let addresses: Vec<_> = addresses.iter().map(Pubkey::to_string).collect();
    body(
        "getMultipleAccounts",
        json!([addresses, {"encoding": "base64"}]),
    )
}

pub fn get_signature_statuses(signatures: &[Signature]) -> Bytes {
    let signatures: Vec<_> = signatures.iter().map(Signature::to_string).collect();
    body("getSignatureStatuses", json!([signatures]))
}

pub fn get_transaction(signature: &Signature) -> Bytes {
    body(
        "getTransaction",
        json!([signature.to_string(), {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]),
    )
}

pub fn get_token_accounts_by_owner(owner: &Pubkey) -> Bytes {
    body(
        "getTokenAccountsByOwner",
        json!([
            owner.to_string(),
            {"programId": spl_token_interface::id().to_string()},
            {"encoding": "jsonParsed"},
        ]),
    )
}

pub fn get_signatures_for_address(address: &Pubkey, limit: usize) -> Bytes {
    body(
        "getSignaturesForAddress",
        json!([address.to_string(), {"limit": limit}]),
    )
}

pub fn get_program_accounts(program: &Pubkey, filters: Value) -> Bytes {
    body(
        "getProgramAccounts",
        json!([program.to_string(), {"encoding": "base64", "filters": filters}]),
    )
}

pub fn scan_filters(world: &World, market: usize) -> Value {
    json!([
        {"dataSize": crate::world::SCAN_ACCOUNT_SIZE},
        {"memcmp": {"offset": 0, "bytes": BASE64_STANDARD.encode(crate::world::SCAN_POSITION), "encoding": "base64"}},
        {"memcmp": {"offset": 8, "bytes": world.markets[market].to_string()}},
    ])
}

pub fn mint_filters(mint: &Pubkey) -> Value {
    json!([
        {"dataSize": spl_token_interface::state::Account::LEN},
        {"memcmp": {"offset": 0, "bytes": mint.to_string()}},
    ])
}

pub fn get_block(slot: u64, transaction_details: &str) -> Bytes {
    body(
        "getBlock",
        json!([slot, {
            "encoding": "json",
            "maxSupportedTransactionVersion": 0,
            "transactionDetails": transaction_details,
            "rewards": false,
        }]),
    )
}

pub fn simulate(transaction: &solana_transaction::Transaction, sig_verify: bool, inner: bool) -> Bytes {
    let wire = BASE64_STANDARD.encode(wincode::serialize(transaction).unwrap());
    body(
        "simulateTransaction",
        json!([wire, {
            "encoding": "base64",
            "sigVerify": sig_verify,
            "replaceRecentBlockhash": !sig_verify,
            "innerInstructions": inner,
        }]),
    )
}

pub fn exec_kind(name: &str) -> TxKind {
    match name {
        "simulate.batch" => TxKind::Batch {
            transfers: BATCH_TRANSFERS,
            fail: false,
        },
        "simulate.cpi" => TxKind::CreateAta(3),
        "simulate.swap" => TxKind::Swap { fail: false },
        "simulate.fail" => TxKind::Batch {
            transfers: BATCH_TRANSFERS,
            fail: true,
        },
        _ => unreachable!("{name} is not an exec type"),
    }
}

/// One exec body for request type `name`. Every other swap body is checked
/// with sigVerify against the real blockhash; the others replace it.
pub fn exec_body(
    rng: &mut ChaCha8Rng,
    world: &World,
    name: &str,
    n: usize,
    blockhash: Hash,
) -> Bytes {
    let kind = exec_kind(name);
    let owner = index(rng, world.active.len());
    let transaction = world.transaction(rng, kind, owner, blockhash);
    let sig_verify = matches!(kind, TxKind::Swap { .. }) && n % 2 == 0;
    simulate(&transaction, sig_verify, matches!(kind, TxKind::CreateAta(_)))
}

/// Draws a key the way a client would: popular accounts are read more often.
struct KeySampler {
    wallets: Zipf,
    token_accounts: Zipf,
    pool_accounts: Zipf,
}

impl KeySampler {
    fn new(world: &World) -> Self {
        Self {
            wallets: Zipf::new(world.wallets.len()),
            token_accounts: Zipf::new(world.token_accounts.len()),
            pool_accounts: Zipf::new(world.pool_accounts.len()),
        }
    }

    fn wallet(&self, rng: &mut ChaCha8Rng, world: &World) -> Pubkey {
        world.wallets[self.wallets.sample(rng)]
    }

    fn token_account(&self, rng: &mut ChaCha8Rng, world: &World) -> Pubkey {
        world.token_accounts[self.token_accounts.sample(rng)].address
    }

    fn pool_account(&self, rng: &mut ChaCha8Rng, world: &World) -> Pubkey {
        world.pool_accounts[self.pool_accounts.sample(rng)]
    }
}

/// Builds the bodies of every request type. `scale` divides the pool sizes.
pub fn build_pools(
    rng: &mut ChaCha8Rng,
    world: &World,
    history: &History,
    scale: usize,
) -> Vec<Vec<Bytes>> {
    let keys = KeySampler::new(world);
    let pool = |len: usize| (len / scale).max(16);
    let tip_blockhash = history.tip.last_blockhash();
    let random_signature = |rng: &mut ChaCha8Rng| {
        let mut bytes = [0u8; 64];
        rng.fill_bytes(&mut bytes);
        Signature::from(bytes)
    };
    REQUEST_TYPES
        .iter()
        .map(|request_type| match request_type.name {
            "getSlot" => vec![body("getSlot", json!([]))],
            "getVersion" => vec![body("getVersion", json!([]))],
            "getLatestBlockhash" => vec![body("getLatestBlockhash", json!([]))],
            "getBalance" => (0..pool(16_384))
                .map(|_| get_balance(&keys.wallet(rng, world)))
                .collect(),
            "getAccountInfo" => (0..pool(16_384))
                .map(|_| {
                    let address = match range(rng, 0, 10) {
                        0..5 => keys.token_account(rng, world),
                        5..8 => keys.wallet(rng, world),
                        _ => keys.pool_account(rng, world),
                    };
                    get_account_info(&address)
                })
                .collect(),
            "getMultipleAccounts" => (0..pool(4_096))
                .map(|_| {
                    let mut addresses = Vec::with_capacity(100);
                    for _ in 0..60 {
                        addresses.push(keys.token_account(rng, world));
                    }
                    for _ in 0..30 {
                        addresses.push(keys.wallet(rng, world));
                    }
                    for _ in 0..10 {
                        addresses.push(keys.pool_account(rng, world));
                    }
                    get_multiple_accounts(&addresses)
                })
                .collect(),
            "getSignatureStatuses" => (0..pool(4_096))
                .map(|_| {
                    let mut signatures = Vec::with_capacity(64);
                    for _ in 0..48 {
                        let transaction = index(rng, history.transactions.len());
                        signatures.push(history.transactions[transaction].signature);
                    }
                    for _ in 0..16 {
                        signatures.push(random_signature(rng));
                    }
                    get_signature_statuses(&signatures)
                })
                .collect(),
            "getTransaction" => (0..pool(4_096))
                .map(|_| {
                    let transaction = &history.transactions[index(rng, history.transactions.len())];
                    get_transaction(&transaction.signature)
                })
                .collect(),
            "getTokenAccountsByOwner" => (0..pool(4_096))
                .map(|_| get_token_accounts_by_owner(&world.wallets[index(rng, world.active.len())]))
                .collect(),
            "getSignaturesForAddress" => (0..pool(4_096))
                .map(|_| {
                    get_signatures_for_address(&world.wallets[index(rng, world.active.len())], 20)
                })
                .collect(),
            "getProgramAccounts.scan" => (0..crate::world::NUM_MARKETS)
                .map(|market| get_program_accounts(&world.scan_program, scan_filters(world, market)))
                .collect(),
            "getProgramAccounts.mint" => (MINT_QUERY_RANK..MINT_QUERY_RANK + 3)
                .map(|rank| {
                    let mint = &world.token_mints[rank.min(world.token_mints.len() - 1)];
                    get_program_accounts(&spl_token_interface::id(), mint_filters(mint))
                })
                .collect(),
            "getTokenAccountsByOwner.whale" => world
                .whales
                .iter()
                .map(get_token_accounts_by_owner)
                .collect(),
            "getBlock.full" => history
                .transaction_slots
                .iter()
                .map(|slot| get_block(*slot, "full"))
                .collect(),
            "getBlock.signatures" => history
                .transaction_slots
                .iter()
                .map(|slot| get_block(*slot, "signatures"))
                .collect(),
            "getSignaturesForAddress.token" => {
                vec![get_signatures_for_address(&spl_token_interface::id(), 1000)]
            }
            name => (0..pool(512))
                .map(|n| exec_body(rng, world, name, n, tip_blockhash))
                .collect(),
        })
        .collect()
}

/// Returns `requests` (type, body) pairs in a seeded order. Each type gets a
/// fixed count, from its class share in `mode` and its weight in the class.
pub fn sequence(
    rng: &mut ChaCha8Rng,
    mode: Mode,
    requests: usize,
    pools: &[Vec<Bytes>],
) -> Vec<(usize, usize)> {
    let mut counts: Vec<usize> = REQUEST_TYPES
        .iter()
        .map(|t| (requests as u64 * mode.class_share(t.class) * t.weight / 100_000) as usize)
        .collect();
    // Give the rounding remainder to the most frequent type.
    let most = (0..counts.len()).max_by_key(|i| counts[*i]).unwrap();
    counts[most] += requests - counts.iter().sum::<usize>();
    let mut sequence: Vec<_> = counts
        .iter()
        .enumerate()
        .flat_map(|(request_type, count)| std::iter::repeat_n((request_type, 0), *count))
        .collect();
    for i in (1..sequence.len()).rev() {
        sequence.swap(i, index(rng, i + 1));
    }
    for (request_type, body) in sequence.iter_mut() {
        *body = index(rng, pools[*request_type].len());
    }
    sequence
}
