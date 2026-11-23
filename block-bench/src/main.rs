//! Benchmark for receiving and replaying Alpenglow blocks.
//!
//! The benchmark builds a seeded chain of Alpenglow blocks from a genesis
//! with many voters, payers and token accounts. Each block has a header,
//! entry batches, a footer with the bank hash and BLS certificates, and the
//! alpentick. Every block then goes through the work a validator does for a
//! block it receives:
//!
//! - A, shred sigverify: dedup and `verify_shred_cpu` on a worker pool with a
//!   shared signature cache, as turbine does;
//! - B, window pass: `Shred::new_from_serialized_shred` and `merkle_root` on
//!   a rayon pool, as the window service does;
//! - C, blockstore insert: `insert_shreds_at_location_handle_duplicate` with
//!   erasure recovery, which also builds the double-Merkle block id;
//! - D, replay: from `Bank::new_from_parent` to `insert_bank_hash`, using the
//!   same calls as `process_single_slot`.
//!
//! `--mode full` times A to D, `--mode shreds` times A to C and
//! `--mode replay` times D only. Genesis, block generation, signing and the
//! setup pass that discovers each block's bank hash are not timed.

#![allow(clippy::arithmetic_side_effects)]

use {
    agave_feature_set::FeatureSet,
    agave_votor_messages::{
        certificate::CertSignature,
        consensus_message::{Block, BlockId},
        reward_certificate::{NUM_SLOTS_FOR_REWARD, NotarRewardCertificate},
        vote::Vote,
        wire::get_vote_payload_to_sign,
    },
    agave_worker_pool::{ResultWorkerPool, WorkerPool, WorkerPoolConfig},
    bitvec::prelude::{BitVec, Lsb0},
    clap::{Arg, ArgMatches, Command, crate_description, crate_name},
    jemallocator::Jemalloc,
    rand::{Rng, SeedableRng, seq::index::sample},
    rand_chacha::ChaCha8Rng,
    rayon::prelude::*,
    solana_account::{Account, AccountSharedData},
    solana_accounts_db::accounts_db::ACCOUNTS_DB_CONFIG_FOR_BENCHMARKS,
    solana_bls_signatures::{
        Signature as BLSSignature, SignatureCompressed as BLSSignatureCompressed,
        SignatureProjective,
    },
    solana_clock::Slot,
    solana_cluster_type::ClusterType,
    solana_compute_budget_interface::ComputeBudgetInstruction,
    solana_entry::{
        block_component::{
            BlockComponent, BlockFinalizationCert, BlockFooterV1, VersionedBlockMarker,
            VotesAggregate,
        },
        entry::{Entry, next_hash},
    },
    solana_epoch_schedule::EpochSchedule,
    solana_fee_calculator::FeeRateGovernor,
    solana_genesis_config::GenesisConfig,
    solana_hash::Hash,
    solana_keypair::Keypair,
    solana_ledger::{
        block_error::BlockError,
        blockstore::{Blockstore, create_new_ledger},
        blockstore_meta::BlockLocation,
        blockstore_metrics::BlockstoreInsertionMetrics,
        blockstore_options::LedgerColumnOptions,
        blockstore_processor::{
            BlockstoreProcessorError, ConfirmationProgress, ConfirmationTiming, ProcessOptions,
            ReplayVerificationWorkerPool, check_chained_block_id, confirm_slot,
            process_single_slot, set_alpenglow_ticks,
        },
        leader_schedule_cache::LeaderScheduleCache,
        shred::{
            self, Payload, ProcessShredsStats, Shred, ShredType, Shredder,
            filter::{RetransmitSender, ShredBatch, ShredRecoveryContext},
        },
        sigverify_shreds::{LruCache, SlotPubkeys, verify_shred_cpu},
    },
    solana_message::Message,
    solana_perf::{
        deduper::Deduper,
        packet::{BytesPacket, Meta, PACKET_DATA_SIZE, PACKETS_PER_BATCH, PacketBatch},
    },
    solana_program_pack::Pack,
    solana_pubkey::Pubkey,
    solana_rent::Rent,
    solana_runtime::{
        bank::Bank,
        bank_forks::BankForks,
        block_component_processor::BlockComponentProcessor,
        genesis_utils::{
            ValidatorVoteKeypairs, create_genesis_config_with_vote_accounts_and_cluster_type,
        },
        installed_scheduler_pool::BankWithScheduler,
        runtime_config::RuntimeConfig,
    },
    solana_runtime_transaction::runtime_transaction::RuntimeTransaction,
    solana_sha256_hasher::hashv,
    solana_signer::Signer,
    solana_signer_store::encode_base2,
    solana_svm_timings::{ExecuteTimingType, ExecuteTimings},
    solana_system_interface::{instruction as system_instruction, program as system_program},
    solana_transaction::{
        Transaction,
        sanitized::{MessageHash, SanitizedTransaction},
        versioned::VersionedTransaction,
    },
    solana_unified_scheduler_pool::DefaultSchedulerPool,
    spl_associated_token_account_interface::{
        address::get_associated_token_address_with_program_id,
        instruction::create_associated_token_account_idempotent,
    },
    spl_token_interface::state::{Account as TokenAccount, AccountState, Mint},
    std::{
        cell::Cell,
        collections::{BTreeMap, HashMap},
        fs,
        hint::black_box,
        num::NonZeroUsize,
        path::{Path, PathBuf},
        sync::{Arc, RwLock},
        thread,
        time::{Duration, Instant},
    },
};

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const LAMPORTS_PER_SOL: u64 = 1_000_000_000;
/// Total stake spread over the voters, about the stake on mainnet.
const TOTAL_STAKE_SOL: u64 = 400_000_000;
const MINT_LAMPORTS: u64 = 1_000_000 * LAMPORTS_PER_SOL;
const PAYER_LAMPORTS: u64 = 10 * LAMPORTS_PER_SOL;
/// Any fixed value; a fixed creation time keeps the bank hashes deterministic.
const GENESIS_CREATION_TIME: i64 = 1_700_000_000;
const NUM_TOKEN_MINTS: usize = 8;
const NUM_TOKEN_2022_MINTS: usize = 2;
const TOKEN_DECIMALS: u8 = 6;
const TOKEN_BALANCE: u64 = 1_000_000_000_000;
/// Share of p-token transfers that go to the hot account of their mint.
const HOT_ACCOUNT_PERCENT: u32 = 10;
const BLOCK_USER_AGENT: &[u8] = b"agave-block-bench";
/// Footer time step, below the 400 ms slot of the default genesis.
const FOOTER_TIME_STEP_NANOS: i64 = 200_000_000;
/// Compute unit limit used when measuring the cost of each transaction class.
const CALIBRATION_CU_LIMIT: u32 = 1_400_000;
// The constants below match turbine/src/sigverify_shreds.rs.
const SIGVERIFY_LRU_CACHE_CAPACITY: usize = 1 << 18;
const DEDUPER_NUM_BITS: u64 = 637_534_199;
const SIGVERIFY_BATCHES_PER_PASS: usize = 128;
/// The window service runs its rayon pool with this minimum length per job.
const WINDOW_MIN_LEN: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Full,
    Shreds,
    Replay,
}

impl Mode {
    fn runs_receive(self) -> bool {
        self != Mode::Replay
    }

    fn runs_replay(self) -> bool {
        self != Mode::Shreds
    }
}

/// Transaction classes. Every transaction also sets a compute unit limit and
/// a compute unit price.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum TxClass {
    /// System transfer.
    Transfer,
    /// p-token TransferChecked between two accounts of the same mint.
    Token,
    /// Token-2022 TransferChecked.
    Token2022,
    /// ATA CreateIdempotent on an existing account, then a p-token transfer.
    Ata,
    /// Memo v3 with a random length in `--memo-bytes`, then a system transfer.
    Memo,
    /// System transfer whose source is not the fee payer: two signatures.
    TwoSigner,
    /// Memo v3 as long as the packet allows. Only added to reach `--cu-target`.
    Heavy,
}

impl TxClass {
    const ALL: [TxClass; 7] = [
        TxClass::Transfer,
        TxClass::Token,
        TxClass::Token2022,
        TxClass::Ata,
        TxClass::Memo,
        TxClass::TwoSigner,
        TxClass::Heavy,
    ];

    fn name(self) -> &'static str {
        match self {
            TxClass::Transfer => "transfer",
            TxClass::Token => "token",
            TxClass::Token2022 => "token2022",
            TxClass::Ata => "ata",
            TxClass::Memo => "memo",
            TxClass::TwoSigner => "two-signer",
            TxClass::Heavy => "heavy",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.name() == name)
    }

    fn num_payers(self) -> usize {
        match self {
            TxClass::TwoSigner => 2,
            _ => 1,
        }
    }
}

/// Mix presets. The weights are percentages of `--txs`.
///
/// `mainnet` follows the measured mainnet shares where this benchmark has a
/// matching class. The design also lists ATA creation for new owners (3%) and
/// v0 transactions through lookup tables (15%); those are not generated yet
/// and their share is folded into `token` and `transfer`.
const MIX_MAINNET: &str = "transfer=13,token=45,token2022=5,ata=20,memo=12,two-signer=5";
const MIX_TRANSFER: &str = "transfer=50,token=50";
/// `cu20m` is the mainnet mix plus heavy memo transactions up to this target.
const CU20M_TARGET: u64 = 20_000_000;

struct Config {
    mode: Mode,
    mix_name: String,
    mix: Vec<(TxClass, u32)>,
    cu_target: u64,
    cu_price: u64,
    slots: u64,
    warmup: u64,
    txs: usize,
    entry_txs: (usize, usize),
    memo_bytes: (usize, usize),
    voters: usize,
    payers: usize,
    drop_data: usize,
    drop_code: usize,
    insert_chunk: usize,
    root_depth: u64,
    final_cert: bool,
    reward_cert: bool,
    handler_threads: usize,
    verify_threads: usize,
    sigverify_threads: usize,
    window_threads: usize,
    seed: u64,
    ledger_dir: Option<PathBuf>,
    deactivate_features: Option<PathBuf>,
    check: bool,
}

fn parse_range(value: &str) -> Result<(usize, usize), String> {
    let (low, high) = match value.split_once('-') {
        Some((low, high)) => (low, high),
        None => (value, value),
    };
    let low: usize = low.parse().map_err(|err| format!("{value}: {err}"))?;
    let high: usize = high.parse().map_err(|err| format!("{value}: {err}"))?;
    if low == 0 || low > high {
        return Err(format!("{value}: expected MIN-MAX with 0 < MIN <= MAX"));
    }
    Ok((low, high))
}

fn parse_mix(value: &str) -> Result<(Vec<(TxClass, u32)>, u64), String> {
    let (spec, cu_target) = match value {
        "mainnet" => (MIX_MAINNET, 0),
        "transfer" => (MIX_TRANSFER, 0),
        "cu20m" => (MIX_MAINNET, CU20M_TARGET),
        spec => (spec, 0),
    };
    let mut mix = Vec::new();
    for item in spec.split(',') {
        let (name, weight) = item
            .split_once('=')
            .ok_or_else(|| format!("{item}: expected CLASS=WEIGHT"))?;
        let class = TxClass::from_name(name)
            .filter(|class| *class != TxClass::Heavy)
            .ok_or_else(|| format!("{name}: unknown transaction class"))?;
        let weight: u32 = weight.parse().map_err(|err| format!("{item}: {err}"))?;
        if weight > 0 {
            mix.push((class, weight));
        }
    }
    if mix.is_empty() {
        return Err(format!("{value}: the mix has no transactions"));
    }
    Ok((mix, cu_target))
}

fn value<T: std::str::FromStr>(matches: &ArgMatches, name: &str) -> T
where
    T::Err: std::fmt::Display,
{
    let value = matches.value_of(name).unwrap();
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid --{name} {value}: {err}"))
}

fn optional_value<T: std::str::FromStr>(matches: &ArgMatches, name: &str) -> Option<T>
where
    T::Err: std::fmt::Display,
{
    matches.is_present(name).then(|| value(matches, name))
}

fn parse_args() -> Config {
    let matches = Command::new(crate_name!())
        .about(crate_description!())
        .arg(
            Arg::new("mode")
                .long("mode")
                .takes_value(true)
                .possible_values(["full", "shreds", "replay"])
                .default_value("full")
                .help("Stages to time: full = A-D, shreds = A-C, replay = D"),
        )
        .arg(
            Arg::new("mix")
                .long("mix")
                .takes_value(true)
                .default_value("mainnet")
                .help(
                    "Transaction mix: mainnet, transfer, cu20m, or a list of CLASS=WEIGHT with \
                     classes transfer, token, token2022, ata, memo, two-signer",
                ),
        )
        .arg(
            Arg::new("cu_target")
                .long("cu-target")
                .takes_value(true)
                .help(
                    "Add heavy memo transactions until the estimated consumed compute units \
                     per block reach this value [default: 20000000 for --mix cu20m, else 0]",
                ),
        )
        .arg(
            Arg::new("cu_price")
                .long("cu-price")
                .takes_value(true)
                .default_value("1000")
                .help("Compute unit price of every transaction, in micro-lamports"),
        )
        .arg(
            Arg::new("slots")
                .long("slots")
                .takes_value(true)
                .default_value("150")
                .help("Measured slots"),
        )
        .arg(
            Arg::new("warmup")
                .long("warmup")
                .takes_value(true)
                .default_value("16")
                .help("Slots processed before the measured slots"),
        )
        .arg(
            Arg::new("txs")
                .long("txs")
                .takes_value(true)
                .default_value("600")
                .help("Transactions per block from the mix, before any --cu-target tail"),
        )
        .arg(
            Arg::new("entry_txs")
                .long("entry-txs")
                .takes_value(true)
                .default_value("1-8")
                .help("Transactions per entry, drawn uniformly from MIN-MAX"),
        )
        .arg(
            Arg::new("memo_bytes")
                .long("memo-bytes")
                .takes_value(true)
                .default_value("200-500")
                .help("Memo length of the memo class, drawn uniformly from MIN-MAX"),
        )
        .arg(
            Arg::new("voters")
                .long("voters")
                .takes_value(true)
                .default_value("1000")
                .help("Staked validators with BLS keys; all of them sign every certificate"),
        )
        .arg(
            Arg::new("payers")
                .long("payers")
                .takes_value(true)
                .default_value("20000")
                .help("Fee payers, each with one p-token and one Token-2022 account"),
        )
        .arg(
            Arg::new("drop_data")
                .long("drop-data")
                .takes_value(true)
                .default_value("2")
                .help("Data shreds dropped per FEC set, recovered by erasure coding"),
        )
        .arg(
            Arg::new("drop_code")
                .long("drop-code")
                .takes_value(true)
                .default_value("2")
                .help("Coding shreds dropped per FEC set"),
        )
        .arg(
            Arg::new("insert_chunk")
                .long("insert-chunk")
                .takes_value(true)
                .default_value("128")
                .help("Shreds per blockstore insert call"),
        )
        .arg(
            Arg::new("root_depth")
                .long("root-depth")
                .takes_value(true)
                .default_value("2")
                .help("After slot S, root slot S minus this value (untimed)"),
        )
        .arg(
            Arg::new("final_cert")
                .long("final-cert")
                .takes_value(true)
                .possible_values(["fast", "none"])
                .default_value("fast")
                .help("Finalization certificate in each footer, for slot S-2"),
        )
        .arg(
            Arg::new("no_reward_cert")
                .long("no-reward-cert")
                .help("Leave out the notarization reward certificate for slot S-8"),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .takes_value(true)
                .help(
                    "Thread count the other thread defaults derive from [default: available \
                     parallelism]",
                ),
        )
        .arg(
            Arg::new("handler_threads")
                .long("handler-threads")
                .takes_value(true)
                .help("Unified scheduler handler threads [default: threads / 4]"),
        )
        .arg(
            Arg::new("verify_threads")
                .long("verify-threads")
                .takes_value(true)
                .help("Replay PoH and signature verification threads [default: threads]"),
        )
        .arg(
            Arg::new("sigverify_threads")
                .long("sigverify-threads")
                .takes_value(true)
                .help("Shred sigverify threads [default: threads]"),
        )
        .arg(
            Arg::new("window_threads")
                .long("window-threads")
                .takes_value(true)
                .help("Window pass threads [default: min(threads, 8)]"),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .takes_value(true)
                .default_value("42")
                .help("Seed for every generated key, transaction and dropped shred"),
        )
        .arg(
            Arg::new("ledger_dir")
                .long("ledger-dir")
                .takes_value(true)
                .help("Directory for the ledgers and accounts [default: a temporary directory]"),
        )
        .arg(
            Arg::new("deactivate_features")
                .long("deactivate-features")
                .takes_value(true)
                .help(
                    "File with feature ids to deactivate, one per line ('#' starts a comment). \
                     Without it every feature is active.",
                ),
        )
        .arg(
            Arg::new("check")
                .long("check")
                .help("Run correctness checks and exit with an error if one fails"),
        )
        .get_matches();

    let threads: usize = optional_value(&matches, "threads").unwrap_or_else(|| {
        thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(1)
    });
    let mix_name = matches.value_of("mix").unwrap().to_string();
    let (mix, preset_cu_target) = parse_mix(&mix_name).unwrap_or_else(|err| panic!("--mix {err}"));
    let entry_txs = parse_range(matches.value_of("entry_txs").unwrap())
        .unwrap_or_else(|err| panic!("--entry-txs {err}"));
    let memo_bytes = parse_range(matches.value_of("memo_bytes").unwrap())
        .unwrap_or_else(|err| panic!("--memo-bytes {err}"));
    let mode = match matches.value_of("mode").unwrap() {
        "full" => Mode::Full,
        "shreds" => Mode::Shreds,
        _ => Mode::Replay,
    };
    let config = Config {
        mode,
        mix_name,
        mix,
        cu_target: optional_value(&matches, "cu_target").unwrap_or(preset_cu_target),
        cu_price: value(&matches, "cu_price"),
        slots: value(&matches, "slots"),
        warmup: value(&matches, "warmup"),
        txs: value(&matches, "txs"),
        entry_txs,
        memo_bytes,
        voters: value(&matches, "voters"),
        payers: value(&matches, "payers"),
        drop_data: value(&matches, "drop_data"),
        drop_code: value(&matches, "drop_code"),
        insert_chunk: value(&matches, "insert_chunk"),
        root_depth: value(&matches, "root_depth"),
        final_cert: matches.value_of("final_cert") == Some("fast"),
        reward_cert: !matches.is_present("no_reward_cert"),
        handler_threads: optional_value(&matches, "handler_threads")
            .unwrap_or((threads / 4).max(1)),
        verify_threads: optional_value(&matches, "verify_threads").unwrap_or(threads),
        sigverify_threads: optional_value(&matches, "sigverify_threads").unwrap_or(threads),
        window_threads: optional_value(&matches, "window_threads").unwrap_or(threads.min(8)),
        seed: value(&matches, "seed"),
        ledger_dir: matches.value_of("ledger_dir").map(PathBuf::from),
        deactivate_features: matches.value_of("deactivate_features").map(PathBuf::from),
        check: matches.is_present("check"),
    };
    assert!(config.slots > 0, "--slots must be at least 1");
    assert!(config.txs > 0, "--txs must be at least 1");
    assert!(config.voters > 0, "--voters must be at least 1");
    assert!(
        config.payers >= 2 * NUM_TOKEN_MINTS,
        "--payers must be at least {}",
        2 * NUM_TOKEN_MINTS
    );
    assert!(config.insert_chunk > 0, "--insert-chunk must be at least 1");
    assert!(config.root_depth > 0, "--root-depth must be at least 1");
    assert!(
        config.drop_data + config.drop_code <= shred::CODING_SHREDS_PER_FEC_BLOCK,
        "dropping more than {} shreds per FEC set prevents recovery",
        shred::CODING_SHREDS_PER_FEC_BLOCK
    );
    for threads in [
        config.handler_threads,
        config.verify_threads,
        config.sigverify_threads,
        config.window_threads,
    ] {
        assert!(threads > 0, "thread counts must be at least 1");
    }
    config
}

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn micros(us: u64) -> f64 {
    us as f64 / 1e6
}

/// Keys and accounts of the generated cluster.
struct World {
    genesis_config: GenesisConfig,
    voters: Vec<ValidatorVoteKeypairs>,
    voter_by_node: HashMap<Pubkey, usize>,
    voter_by_vote_account: HashMap<Pubkey, usize>,
    payers: Vec<Keypair>,
    token_mints: Vec<Pubkey>,
    token_2022_mints: Vec<Pubkey>,
    /// p-token account of payer `i`, for mint `i % NUM_TOKEN_MINTS`.
    token_accounts: Vec<Pubkey>,
    /// Token-2022 account of payer `i`, for mint `i % NUM_TOKEN_2022_MINTS`.
    token_2022_accounts: Vec<Pubkey>,
    /// One p-token account per mint that many transfers go to.
    hot_accounts: Vec<Pubkey>,
}

fn read_feature_ids(path: &Path) -> Vec<Pubkey> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    text.lines()
        .map(|line| line.split('#').next().unwrap().trim())
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.parse()
                .unwrap_or_else(|err| panic!("invalid feature id {line}: {err}"))
        })
        .collect()
}

fn token_account_data(mint: Pubkey, owner: Pubkey) -> Vec<u8> {
    let account = TokenAccount {
        mint,
        owner,
        amount: TOKEN_BALANCE,
        state: AccountState::Initialized,
        ..TokenAccount::default()
    };
    let mut data = vec![0; TokenAccount::LEN];
    account.pack_into_slice(&mut data);
    data
}

fn mint_data(supply: u64) -> Vec<u8> {
    let mint = Mint {
        supply,
        decimals: TOKEN_DECIMALS,
        is_initialized: true,
        ..Mint::default()
    };
    let mut data = vec![0; Mint::LEN];
    mint.pack_into_slice(&mut data);
    data
}

fn build_world(config: &Config) -> World {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let voters: Vec<_> = (0..config.voters)
        .map(|_| {
            let mut keypair = || Keypair::new_from_array(rng.random());
            ValidatorVoteKeypairs::new(keypair(), keypair(), keypair())
        })
        .collect();
    // Zipf-like stake skew (s = 1), as on mainnet where a few validators hold
    // a large share of the stake.
    let harmonic: f64 = (1..=config.voters).map(|i| 1.0 / i as f64).sum();
    let stakes: Vec<u64> = (1..=config.voters)
        .map(|i| {
            let sol = TOTAL_STAKE_SOL as f64 / harmonic / i as f64;
            (sol as u64).max(1) * LAMPORTS_PER_SOL
        })
        .collect();

    let mut feature_set = FeatureSet::all_enabled();
    if let Some(path) = &config.deactivate_features {
        for id in read_feature_ids(path) {
            assert_ne!(
                id,
                agave_feature_set::alpenglow::id(),
                "this benchmark needs Alpenglow"
            );
            feature_set.deactivate(&id);
        }
    }
    let mut genesis_config = create_genesis_config_with_vote_accounts_and_cluster_type(
        MINT_LAMPORTS,
        &voters,
        stakes,
        ClusterType::Development,
        feature_set,
    )
    .genesis_config;
    // The helper sets zero fees, free rent and the current time; use mainnet
    // fee and rent rules and a fixed creation time instead. Without warmup an
    // epoch is 432,000 slots, so no run crosses an epoch boundary.
    genesis_config.creation_time = GENESIS_CREATION_TIME;
    genesis_config.fee_rate_governor = FeeRateGovernor::default();
    genesis_config.rent = Rent::default();
    genesis_config.epoch_schedule = EpochSchedule::without_warmup();
    let rent = genesis_config.rent.clone();

    let programs = solana_program_binaries::spl_programs(&rent)
        .into_iter()
        .chain(solana_program_binaries::core_bpf_programs(&rent, |_| true));
    for (pubkey, account) in programs {
        genesis_config.add_account(pubkey, account);
    }

    let payers: Vec<_> = (0..config.payers)
        .map(|_| Keypair::new_from_array(rng.random()))
        .collect();
    let token_mints: Vec<Pubkey> = (0..NUM_TOKEN_MINTS)
        .map(|_| Pubkey::new_from_array(rng.random()))
        .collect();
    let token_2022_mints: Vec<Pubkey> = (0..NUM_TOKEN_2022_MINTS)
        .map(|_| Pubkey::new_from_array(rng.random()))
        .collect();
    let hot_owners: Vec<Pubkey> = (0..NUM_TOKEN_MINTS)
        .map(|_| Pubkey::new_from_array(rng.random()))
        .collect();
    let token_program = spl_token_interface::id();
    let token_2022_program = spl_token_2022_interface::id();
    let token_accounts: Vec<Pubkey> = payers
        .par_iter()
        .enumerate()
        .map(|(i, payer)| {
            get_associated_token_address_with_program_id(
                &payer.pubkey(),
                &token_mints[i % NUM_TOKEN_MINTS],
                &token_program,
            )
        })
        .collect();
    let token_2022_accounts: Vec<Pubkey> = payers
        .par_iter()
        .enumerate()
        .map(|(i, payer)| {
            get_associated_token_address_with_program_id(
                &payer.pubkey(),
                &token_2022_mints[i % NUM_TOKEN_2022_MINTS],
                &token_2022_program,
            )
        })
        .collect();
    let hot_accounts: Vec<Pubkey> = hot_owners
        .iter()
        .zip(&token_mints)
        .map(|(owner, mint)| get_associated_token_address_with_program_id(owner, mint, &token_program))
        .collect();

    // Token-2022 accounts without extensions have the same layout as p-token
    // accounts, so both programs use the same account and mint data.
    let token_account_lamports = rent.minimum_balance(TokenAccount::LEN);
    let mint_lamports = rent.minimum_balance(Mint::LEN);
    let mint_supply = TOKEN_BALANCE * config.payers as u64;
    for (mints, program) in [
        (&token_mints, token_program),
        (&token_2022_mints, token_2022_program),
    ] {
        for mint in mints.iter() {
            genesis_config.add_account(
                *mint,
                AccountSharedData::from(Account {
                    lamports: mint_lamports,
                    data: mint_data(mint_supply),
                    owner: program,
                    executable: false,
                    rent_epoch: 0,
                }),
            );
        }
    }
    let token_account = |mint: Pubkey, owner: Pubkey, program: Pubkey| {
        AccountSharedData::from(Account {
            lamports: token_account_lamports,
            data: token_account_data(mint, owner),
            owner: program,
            executable: false,
            rent_epoch: 0,
        })
    };
    for (i, payer) in payers.iter().enumerate() {
        genesis_config.add_account(
            payer.pubkey(),
            AccountSharedData::empty(PAYER_LAMPORTS, &system_program::id()),
        );
        genesis_config.add_account(
            token_accounts[i],
            token_account(
                token_mints[i % NUM_TOKEN_MINTS],
                payer.pubkey(),
                token_program,
            ),
        );
        genesis_config.add_account(
            token_2022_accounts[i],
            token_account(
                token_2022_mints[i % NUM_TOKEN_2022_MINTS],
                payer.pubkey(),
                token_2022_program,
            ),
        );
    }
    for (i, hot_account) in hot_accounts.iter().enumerate() {
        genesis_config.add_account(
            *hot_account,
            token_account(token_mints[i], hot_owners[i], token_program),
        );
    }

    let voter_by_node = voters
        .iter()
        .enumerate()
        .map(|(i, voter)| (voter.node_keypair.pubkey(), i))
        .collect();
    let voter_by_vote_account = voters
        .iter()
        .enumerate()
        .map(|(i, voter)| (voter.vote_keypair.pubkey(), i))
        .collect();
    World {
        genesis_config,
        voters,
        voter_by_node,
        voter_by_vote_account,
        payers,
        token_mints,
        token_2022_mints,
        token_accounts,
        token_2022_accounts,
        hot_accounts,
    }
}

/// A bank chain with its blockstore, as replay sees it.
struct Chain {
    bank_forks: Arc<RwLock<BankForks>>,
    leader_schedule_cache: Arc<LeaderScheduleCache>,
    blockstore: Arc<Blockstore>,
}

fn create_ledger(path: &Path, genesis_config: &GenesisConfig) -> Blockstore {
    create_new_ledger(
        path,
        genesis_config,
        u64::MAX,
        LedgerColumnOptions::default(),
    )
    .unwrap_or_else(|err| panic!("failed to create ledger {}: {err}", path.display()));
    Blockstore::open(path).unwrap()
}

/// Builds bank 0 the way startup does for a new ledger: its block id comes
/// from the slot 0 shreds, then it is frozen.
fn new_chain(
    genesis_config: &GenesisConfig,
    blockstore: Arc<Blockstore>,
    accounts_dir: PathBuf,
    handler_threads: usize,
) -> Chain {
    let bank0 = Bank::new_from_genesis(
        genesis_config,
        Arc::new(RuntimeConfig::default()),
        vec![accounts_dir],
        None,
        ACCOUNTS_DB_CONFIG_FOR_BENCHMARKS,
        None,
        None,
        Arc::default(),
        None,
        None,
    );
    let bank_forks = BankForks::new_rw_arc(bank0);
    bank_forks
        .write()
        .unwrap()
        .install_scheduler_pool(DefaultSchedulerPool::new(
            Some(handler_threads),
            None,
            None,
            None,
            None,
        ));
    let bank0 = bank_forks.read().unwrap().root_bank();
    let migration_status = bank_forks.read().unwrap().migration_status();
    let block_id = blockstore
        .get_block_id(0, &migration_status)
        .unwrap()
        .expect("slot 0 has a block id");
    bank0.set_block_id(Some(BlockId::from(block_id)));
    bank0.freeze();
    let leader_schedule_cache = Arc::new(LeaderScheduleCache::new_from_bank(&bank0));
    Chain {
        bank_forks,
        leader_schedule_cache,
        blockstore,
    }
}

impl Chain {
    fn bank(&self, slot: Slot) -> Arc<Bank> {
        self.bank_forks.read().unwrap().get(slot).unwrap()
    }

    fn new_bank(&self, parent: &Arc<Bank>, slot: Slot) -> BankWithScheduler {
        let leader = self
            .leader_schedule_cache
            .slot_leader_at(slot, Some(parent))
            .unwrap();
        let bank = Bank::new_from_parent(parent.clone(), leader, slot);
        let migration_status = self.bank_forks.read().unwrap().migration_status();
        set_alpenglow_ticks(&bank, &migration_status);
        self.bank_forks.write().unwrap().insert(bank)
    }

    /// Roots `slot`, untimed. Without roots every bank stays in `BankForks`
    /// and the accounts index keeps one more slot per account per block.
    fn set_root(&self, slot: Slot) -> Arc<Bank> {
        self.bank_forks.write().unwrap().set_root(slot, None, None);
        let root_bank = self.bank_forks.read().unwrap().root_bank();
        self.leader_schedule_cache.set_root(&root_bank);
        root_bank
    }
}

/// Stage D timings, summed over slots, in microseconds.
#[derive(Default)]
struct ReplayStats {
    bank_new_us: u64,
    confirm_us: u64,
    fetch_us: u64,
    confirmation_us: u64,
    replay_us: u64,
    poh_verify_us: u64,
    sig_verify_us: u64,
    drain_us: u64,
    freeze_us: u64,
    execute_timings: ExecuteTimings,
}

/// How replay finishes a slot.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Finish {
    /// Set the block id from the blockstore, check the footer bank hash and
    /// record the bank hash, as `process_single_slot` does.
    Verify,
    /// Only freeze; the caller sets the block id. Used while discovering bank
    /// hashes, when the footer carries a placeholder hash.
    Discover,
}

/// Shreds block components the way `StandardBroadcastRun` does: one call per
/// component, chained to the last FEC set of the previous component.
#[derive(Clone)]
struct ComponentShredder {
    slot: Slot,
    parent_slot: Slot,
    shred_version: u16,
    chained_merkle_root: Hash,
    next_shred_index: u32,
    next_code_index: u32,
}

impl ComponentShredder {
    fn shred(
        &mut self,
        keypair: &Keypair,
        component: &BlockComponent,
        is_last_in_slot: bool,
    ) -> Vec<Shred> {
        let shreds = Shredder::new(self.slot, self.parent_slot, 0, self.shred_version)
            .unwrap()
            .make_merkle_shreds_from_component(
                keypair,
                component,
                is_last_in_slot,
                self.chained_merkle_root,
                self.next_shred_index,
                self.next_code_index,
                &mut ProcessShredsStats::default(),
            );
        for shred in &shreds {
            let next_index = match shred.shred_type() {
                ShredType::Code => &mut self.next_code_index,
                ShredType::Data => &mut self.next_shred_index,
            };
            *next_index = (*next_index).max(shred.index() + 1);
        }
        if let Some(shred) = shreds.iter().max_by_key(|shred| shred.fec_set_index()) {
            self.chained_merkle_root = shred.merkle_root().unwrap();
        }
        shreds
    }
}

/// One generated transaction before signing.
struct TxPlan {
    class: TxClass,
    payer: usize,
    /// Second signer of `TwoSigner`.
    other: usize,
    /// Destination payer of the transfer.
    destination: usize,
    hot: bool,
    amount: u64,
    memo: Vec<u8>,
}

/// Per-class compute units, measured once at setup by simulation.
struct Calibration {
    units: BTreeMap<TxClass, u64>,
    heavy_memo_len: usize,
}

impl Calibration {
    fn cu_limit(&self, class: TxClass) -> u32 {
        // Give each class 25% more than it used, plus room for the memo
        // length to vary.
        let units = self.units.get(&class).copied().unwrap_or(200_000);
        (units + units / 4 + 1_000).min(u64::from(CALIBRATION_CU_LIMIT)) as u32
    }
}

struct Generator<'a> {
    config: &'a Config,
    world: &'a World,
}

impl Generator<'_> {
    fn memo(&self, rng: &mut ChaCha8Rng, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.random_range(b'a'..=b'z')).collect()
    }

    fn build(&self, plan: &TxPlan, cu_limit: u32, recent_blockhash: Hash) -> VersionedTransaction {
        let world = self.world;
        let payer = &world.payers[plan.payer];
        let destination = world.payers[plan.destination].pubkey();
        let mut instructions = vec![
            ComputeBudgetInstruction::set_compute_unit_limit(cu_limit),
            ComputeBudgetInstruction::set_compute_unit_price(self.config.cu_price),
        ];
        let token_transfer = |program: &Pubkey,
                              accounts: &[Pubkey],
                              mints: &[Pubkey],
                              destination: Pubkey| {
            let mint = mints[plan.payer % mints.len()];
            spl_token_interface::instruction::transfer_checked(
                program,
                &accounts[plan.payer],
                &mint,
                &destination,
                &payer.pubkey(),
                &[],
                plan.amount,
                TOKEN_DECIMALS,
            )
            .unwrap()
        };
        let token_destination = || {
            if plan.hot {
                world.hot_accounts[plan.payer % NUM_TOKEN_MINTS]
            } else {
                world.token_accounts[plan.destination]
            }
        };
        let mut signers = vec![payer];
        match plan.class {
            TxClass::Transfer => instructions.push(system_instruction::transfer(
                &payer.pubkey(),
                &destination,
                plan.amount,
            )),
            TxClass::Token => instructions.push(token_transfer(
                &spl_token_interface::id(),
                &world.token_accounts,
                &world.token_mints,
                token_destination(),
            )),
            TxClass::Token2022 => {
                let mint = world.token_2022_mints[plan.payer % NUM_TOKEN_2022_MINTS];
                instructions.push(
                    spl_token_2022_interface::instruction::transfer_checked(
                        &spl_token_2022_interface::id(),
                        &world.token_2022_accounts[plan.payer],
                        &mint,
                        &world.token_2022_accounts[plan.destination],
                        &payer.pubkey(),
                        &[],
                        plan.amount,
                        TOKEN_DECIMALS,
                    )
                    .unwrap(),
                )
            }
            TxClass::Ata => {
                instructions.push(create_associated_token_account_idempotent(
                    &payer.pubkey(),
                    &payer.pubkey(),
                    &world.token_mints[plan.payer % NUM_TOKEN_MINTS],
                    &spl_token_interface::id(),
                ));
                instructions.push(token_transfer(
                    &spl_token_interface::id(),
                    &world.token_accounts,
                    &world.token_mints,
                    token_destination(),
                ));
            }
            TxClass::Memo => {
                instructions.push(spl_memo_interface::instruction::build_memo(
                    &spl_memo_interface::v3::id(),
                    &plan.memo,
                    &[],
                ));
                instructions.push(system_instruction::transfer(
                    &payer.pubkey(),
                    &destination,
                    plan.amount,
                ));
            }
            TxClass::TwoSigner => {
                let other = &world.payers[plan.other];
                instructions.push(system_instruction::transfer(
                    &other.pubkey(),
                    &destination,
                    plan.amount,
                ));
                signers.push(other);
            }
            TxClass::Heavy => instructions.push(spl_memo_interface::instruction::build_memo(
                &spl_memo_interface::v3::id(),
                &plan.memo,
                &[],
            )),
        }
        let message = Message::new(&instructions, Some(&payer.pubkey()));
        VersionedTransaction::from(Transaction::new(&signers, message, recent_blockhash))
    }

    /// Picks a destination payer that holds the same mint as `payer` for the
    /// given number of mints.
    fn same_mint_destination(&self, rng: &mut ChaCha8Rng, payer: usize, num_mints: usize) -> usize {
        let payers = self.config.payers - self.config.payers % num_mints;
        let destination = rng.random_range(0..payers / num_mints) * num_mints + payer % num_mints;
        if destination == payer {
            (destination + num_mints) % payers
        } else {
            destination
        }
    }

    fn plan_class(&self, rng: &mut ChaCha8Rng, class: TxClass, payer: usize, other: usize) -> TxPlan {
        let config = self.config;
        let destination = match class {
            TxClass::Token | TxClass::Ata => self.same_mint_destination(rng, payer, NUM_TOKEN_MINTS),
            TxClass::Token2022 => self.same_mint_destination(rng, payer, NUM_TOKEN_2022_MINTS),
            _ => {
                let destination = rng.random_range(0..config.payers);
                if destination == payer {
                    (destination + 1) % config.payers
                } else {
                    destination
                }
            }
        };
        let hot = matches!(class, TxClass::Token | TxClass::Ata)
            && rng.random_range(0..100) < HOT_ACCOUNT_PERCENT;
        let memo = match class {
            TxClass::Memo => {
                let len = rng.random_range(config.memo_bytes.0..=config.memo_bytes.1);
                self.memo(rng, len)
            }
            _ => Vec::new(),
        };
        TxPlan {
            class,
            payer,
            other,
            destination,
            hot,
            amount: rng.random_range(1..1_000),
            memo,
        }
    }

    /// Plans the transactions of one block: `--txs` from the mix, then heavy
    /// memo transactions until the estimated compute units reach
    /// `--cu-target`.
    fn plan_block(&self, rng: &mut ChaCha8Rng, calibration: &Calibration) -> Vec<TxPlan> {
        let config = self.config;
        let total_weight: u32 = config.mix.iter().map(|(_, weight)| weight).sum();
        let mut classes: Vec<TxClass> = (0..config.txs)
            .map(|_| {
                let mut pick = rng.random_range(0..total_weight);
                for (class, weight) in &config.mix {
                    if pick < *weight {
                        return *class;
                    }
                    pick -= weight;
                }
                unreachable!()
            })
            .collect();
        if config.cu_target > 0 {
            let estimated: u64 = classes
                .iter()
                .map(|class| calibration.units[class])
                .sum();
            let heavy_units = calibration.units[&TxClass::Heavy].max(1);
            let heavy = config.cu_target.saturating_sub(estimated).div_ceil(heavy_units);
            classes.extend((0..heavy).map(|_| TxClass::Heavy));
        }
        let needed: usize = classes.iter().map(|class| class.num_payers()).sum();
        assert!(
            needed <= config.payers,
            "a block needs {needed} distinct payers, raise --payers"
        );
        // Each payer signs at most one transaction per block.
        let mut payers = sample(rng, config.payers, needed).into_iter();
        classes
            .into_iter()
            .map(|class| {
                let payer = payers.next().unwrap();
                let other = if class.num_payers() == 2 {
                    payers.next().unwrap()
                } else {
                    payer
                };
                let mut plan = self.plan_class(rng, class, payer, other);
                if class == TxClass::Heavy {
                    plan.memo = self.memo(rng, calibration.heavy_memo_len);
                }
                plan
            })
            .collect()
    }

    /// Measures the compute units of each class by simulating one transaction
    /// of each on `bank`. Programs deployed at genesis are only visible from
    /// slot 1, so `bank` is a child of bank 0.
    fn calibrate(&self, bank: &Bank) -> Calibration {
        let config = self.config;
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        rng.set_stream(u64::MAX);
        let blockhash = bank.last_blockhash();

        // The longest memo that keeps the transaction within a packet.
        let mut plan = self.plan_class(&mut rng, TxClass::Heavy, 0, 0);
        let base_size = wincode_size(&self.build(&plan, CALIBRATION_CU_LIMIT, blockhash));
        // The memo length prefix grows from 1 to 2 bytes past 127 bytes.
        let heavy_memo_len = PACKET_DATA_SIZE - base_size - 1;
        plan.memo = self.memo(&mut rng, heavy_memo_len);
        assert!(wincode_size(&self.build(&plan, CALIBRATION_CU_LIMIT, blockhash)) <= PACKET_DATA_SIZE);

        let mut units = BTreeMap::new();
        for class in TxClass::ALL {
            let mut plan = self.plan_class(&mut rng, class, 1, 2);
            match class {
                TxClass::Memo => plan.memo = self.memo(&mut rng, config.memo_bytes.1),
                TxClass::Heavy => plan.memo = self.memo(&mut rng, heavy_memo_len),
                _ => (),
            }
            let transaction = self.build(&plan, CALIBRATION_CU_LIMIT, blockhash);
            let size = wincode_size(&transaction);
            let transaction = RuntimeTransaction::<SanitizedTransaction>::try_create(
                transaction,
                MessageHash::Compute,
                bank,
                bank.get_reserved_account_keys(),
            )
            .unwrap();
            let result = bank.simulate_transaction_unchecked(&transaction, false);
            if let Err(err) = &result.result {
                panic!(
                    "calibration transaction of class {} failed: {err}, logs: {:?}",
                    class.name(),
                    result.logs
                );
            }
            println!(
                "CALIBRATION class={} cu={} size={size}",
                class.name(),
                result.units_consumed
            );
            units.insert(class, result.units_consumed);
        }
        Calibration {
            units,
            heavy_memo_len,
        }
    }
}

fn wincode_size(transaction: &VersionedTransaction) -> usize {
    wincode::serialized_size(transaction).unwrap() as usize
}

/// One block in its final form, with what the measured pass needs.
struct BlockData {
    slot: Slot,
    leader: Pubkey,
    shreds: Vec<Shred>,
    /// Shreds dropped on the way, recovered by erasure coding in stage C.
    dropped: Vec<bool>,
    num_dropped_data: usize,
    block_id: Hash,
    bank_hash: Hash,
    num_txs: usize,
    num_sigs: usize,
    num_entries: usize,
    num_batches: usize,
    num_fec_sets: usize,
    entry_bytes: usize,
    /// Sum of the calibrated compute units of the block's transactions.
    estimated_cu: u64,
    /// Block cost from the cost tracker after replay.
    cost_units: u64,
}

/// Aggregates the notarization votes of every voter, in rank order, for one
/// block. Fast finalization uses the same votes.
fn aggregate_notar_votes(
    world: &World,
    rank_signers: &[usize],
    block: Block,
    shred_version: u16,
) -> BLSSignature {
    let payload = get_vote_payload_to_sign(Vote::new_notarization_vote(block), shred_version);
    let signatures: Vec<SignatureProjective> = rank_signers
        .par_iter()
        .map(|voter| world.voters[*voter].bls_keypair.sign(&payload))
        .collect();
    let mut aggregate = SignatureProjective::identity();
    aggregate.aggregate_with(signatures.iter()).unwrap();
    BLSSignature::from(aggregate)
}

/// Picks the shreds lost on the way: `--drop-data` data shreds and
/// `--drop-code` coding shreds of every FEC set.
fn pick_dropped(config: &Config, slot: Slot, shreds: &[Shred]) -> (Vec<bool>, usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    rng.set_stream(slot | 1 << 63);
    let mut fec_sets: BTreeMap<u32, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    for (i, shred) in shreds.iter().enumerate() {
        let (data, code) = fec_sets.entry(shred.fec_set_index()).or_default();
        if shred.is_data() {
            data.push(i);
        } else {
            code.push(i);
        }
    }
    let mut dropped = vec![false; shreds.len()];
    let mut num_dropped_data = 0;
    for (data, code) in fec_sets.values() {
        for (indices, count) in [(data, config.drop_data), (code, config.drop_code)] {
            let count = count.min(indices.len());
            for pick in sample(&mut rng, indices.len(), count) {
                dropped[indices[pick]] = true;
            }
        }
        num_dropped_data += config.drop_data.min(data.len());
    }
    (dropped, num_dropped_data)
}

/// Builds every block. Each block is first replayed with a placeholder
/// footer hash to learn its bank hash, which does not depend on the footer
/// hash or the block id. Then the footer and alpentick are shredded again
/// with the real hash, and the final shreds give the block id that the next
/// header and later certificates refer to.
#[allow(clippy::too_many_arguments)]
fn generate_blocks(
    config: &Config,
    world: &World,
    generation: &Chain,
    final_blockstore: &Blockstore,
    replay_verification_worker_pool: &ReplayVerificationWorkerPool,
    shred_version: u16,
    calibration: &Calibration,
) -> Vec<BlockData> {
    let generator = Generator { config, world };
    let migration_status = generation.bank_forks.read().unwrap().migration_status();
    let bank0 = generation.bank(0);
    let rank_map = bank0
        .epoch_stakes_from_slot(0)
        .expect("epoch stakes for slot 0")
        .bls_pubkey_to_rank_map();
    let rank_signers: Vec<usize> = (0..rank_map.len())
        .map(|rank| {
            let entry = rank_map.get_pubkey_stake_entry(rank).unwrap();
            world.voter_by_vote_account[&entry.vote_account_pubkey]
        })
        .collect();
    let bitmap = encode_base2(&BitVec::<u8, Lsb0>::repeat(true, rank_signers.len())).unwrap();

    let mut block_ids = vec![bank0.block_id().unwrap().to_hash()];
    let slot0_last_index = generation
        .blockstore
        .meta(0)
        .unwrap()
        .and_then(|meta| meta.last_index)
        .expect("slot 0 is full");
    let slot0_last_shred = generation
        .blockstore
        .get_data_shred(0, slot0_last_index)
        .unwrap()
        .expect("slot 0 last shred");
    let mut last_fec_roots = vec![shred::wire::get_merkle_root(&slot0_last_shred).unwrap()];
    let mut notar_votes: Vec<Option<BLSSignature>> = vec![None];
    let batch_target_bytes = 2 * shred::get_data_shred_bytes_per_batch_typical() as usize;
    let last_slot = config.warmup + config.slots;
    let mut blocks = Vec::with_capacity(last_slot as usize);

    for slot in 1..=last_slot {
        let parent = generation.bank(slot - 1);
        let bank = generation.new_bank(&parent, slot);
        let leader = generation
            .leader_schedule_cache
            .slot_leader_at(slot, Some(&parent))
            .unwrap()
            .id;
        let keypair = &world.voters[world.voter_by_node[&leader]].node_keypair;

        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        rng.set_stream(slot);
        let plans = generator.plan_block(&mut rng, calibration);
        let blockhash = parent.last_blockhash();
        let transactions: Vec<VersionedTransaction> = plans
            .par_iter()
            .map(|plan| generator.build(plan, calibration.cu_limit(plan.class), blockhash))
            .collect();
        let num_txs = transactions.len();
        let estimated_cu = plans
            .iter()
            .map(|plan| calibration.units[&plan.class])
            .sum();
        let num_sigs = transactions.iter().map(|tx| tx.signatures.len()).sum();

        let mut entries = Vec::new();
        let mut last_hash = blockhash;
        let mut transactions = transactions.into_iter();
        loop {
            let count = rng.random_range(config.entry_txs.0..=config.entry_txs.1);
            let chunk: Vec<_> = transactions.by_ref().take(count).collect();
            if chunk.is_empty() {
                break;
            }
            last_hash = next_hash(&last_hash, 1, &chunk);
            entries.push(Entry {
                num_hashes: 1,
                hash: last_hash,
                transactions: chunk,
            });
        }
        let num_entries = entries.len() + 1;
        // Pack entries the way broadcast does, into batches of at least two
        // full FEC sets of data.
        let mut batches = Vec::new();
        let mut batch = Vec::new();
        let mut batch_bytes = 0;
        let mut entry_bytes = 0;
        for entry in entries {
            let size = wincode::serialized_size(&entry).unwrap() as usize;
            batch_bytes += size;
            entry_bytes += size;
            batch.push(entry);
            if batch_bytes >= batch_target_bytes {
                batches.push(std::mem::take(&mut batch));
                batch_bytes = 0;
            }
        }
        if !batch.is_empty() {
            batches.push(batch);
        }
        let num_batches = batches.len();
        let alpentick = Entry {
            num_hashes: 1,
            hash: next_hash(&last_hash, 1, &[]),
            transactions: vec![],
        };

        let parent_time_nanos = parent
            .get_nanosecond_clock()
            .unwrap_or_else(|| bank.clock().unix_timestamp.saturating_mul(1_000_000_000));
        let (lower, upper) = BlockComponentProcessor::nanosecond_time_bounds(
            parent_time_nanos,
            bank.slot_range_duration_nanos(slot - 1 + 1, slot),
        );
        let block_producer_time_nanos = (parent_time_nanos + FOOTER_TIME_STEP_NANOS)
            .clamp(lower, upper)
            .try_into()
            .unwrap();
        let notar_reward_cert = (config.reward_cert && slot > NUM_SLOTS_FOR_REWARD).then(|| {
            let reward_slot = slot - NUM_SLOTS_FOR_REWARD;
            let signature = notar_votes[reward_slot as usize].as_ref().unwrap();
            NotarRewardCertificate::try_new(
                reward_slot,
                BlockId::from(block_ids[reward_slot as usize]),
                BLSSignatureCompressed::try_from(signature).unwrap(),
                bitmap.clone(),
            )
            .unwrap()
        });
        let block_final_cert = (config.final_cert && slot > 2).then(|| {
            let final_slot = slot - 2;
            BlockFinalizationCert {
                slot: final_slot,
                block_id: BlockId::from(block_ids[final_slot as usize]),
                final_aggregate: VotesAggregate::from_cert_signature(CertSignature {
                    signature: notar_votes[final_slot as usize].unwrap(),
                    bitmap: bitmap.clone(),
                }),
                notar_aggregate: None,
            }
        });
        let footer = |bank_hash| {
            BlockComponent::BlockMarker(VersionedBlockMarker::from_block_footer(BlockFooterV1 {
                bank_hash,
                block_producer_time_nanos,
                block_user_agent: BLOCK_USER_AGENT.to_vec(),
                block_final_cert: block_final_cert.clone(),
                skip_reward_cert: None,
                notar_reward_cert: notar_reward_cert.clone(),
            }))
        };

        let mut shredder = ComponentShredder {
            slot,
            parent_slot: slot - 1,
            shred_version,
            chained_merkle_root: last_fec_roots[slot as usize - 1],
            next_shred_index: 0,
            next_code_index: 0,
        };
        let header = BlockComponent::new_block_header(
            slot - 1,
            BlockId::from(block_ids[slot as usize - 1]),
        );
        let mut shreds = shredder.shred(keypair, &header, false);
        for batch in batches {
            shreds.extend(shredder.shred(keypair, &BlockComponent::EntryBatch(batch), false));
        }
        let shred_tail = |bank_hash| {
            let mut shredder = shredder.clone();
            let mut shreds = shredder.shred(keypair, &footer(bank_hash), false);
            shreds.extend(shredder.shred(
                keypair,
                &BlockComponent::EntryBatch(vec![alpentick.clone()]),
                true,
            ));
            (shreds, shredder.chained_merkle_root)
        };

        let (placeholder_tail, _) = shred_tail(Hash::default());
        let placeholder: Vec<Shred> = shreds.iter().cloned().chain(placeholder_tail).collect();
        generation.blockstore.insert_shreds(placeholder, true).unwrap();
        let mut stats = ReplayStats::default();
        replay_bank(
            generation,
            replay_verification_worker_pool,
            shred_version,
            &parent,
            &bank,
            Finish::Discover,
            &mut stats,
        )
        .unwrap_or_else(|err| panic!("generated slot {slot} failed to replay: {err}"));
        let bank_hash = bank.hash();
        let cost_units = bank.read_cost_tracker().unwrap().block_cost();

        let (tail, last_fec_root) = shred_tail(bank_hash);
        shreds.extend(tail);
        final_blockstore.insert_shreds(shreds.clone(), true).unwrap();
        let block_id = final_blockstore
            .get_block_id(slot, &migration_status)
            .unwrap()
            .expect("a full block has a block id");
        bank.set_block_id(Some(BlockId::from(block_id)));
        notar_votes.push(Some(aggregate_notar_votes(
            world,
            &rank_signers,
            Block {
                slot,
                block_id: BlockId::from(block_id),
            },
            shred_version,
        )));
        block_ids.push(block_id);
        last_fec_roots.push(last_fec_root);
        if slot > config.root_depth {
            generation.set_root(slot - config.root_depth);
        }

        let (dropped, num_dropped_data) = pick_dropped(config, slot, &shreds);
        let mut fec_sets: Vec<u32> = shreds.iter().map(Shred::fec_set_index).collect();
        fec_sets.sort_unstable();
        fec_sets.dedup();
        blocks.push(BlockData {
            slot,
            leader,
            dropped,
            num_dropped_data,
            block_id,
            bank_hash,
            num_txs,
            num_sigs,
            num_entries,
            num_batches,
            num_fec_sets: fec_sets.len(),
            entry_bytes,
            estimated_cu,
            cost_units,
            shreds,
        });
    }
    blocks
}

struct SigverifyJob {
    index: usize,
    batch: PacketBatch,
    slot_leaders: Arc<SlotPubkeys>,
}

struct SigverifyOutput {
    index: usize,
    batch: PacketBatch,
    num_duplicates: usize,
    num_failed: usize,
}

/// Same work as `verify_batch` in turbine/src/sigverify_shreds.rs: dedup,
/// then `verify_shred_cpu` against the slot leader with the shared cache.
fn sigverify_batch(
    job: SigverifyJob,
    deduper: &Deduper<2, [u8]>,
    cache: &RwLock<LruCache>,
) -> SigverifyOutput {
    let SigverifyJob {
        index,
        mut batch,
        slot_leaders,
    } = job;
    let mut num_duplicates = 0;
    for packet in batch.iter_mut() {
        let duplicate = shred::wire::get_shred_without_retransmitter_signature(packet)
            .map(|shred| deduper.dedup(shred))
            .unwrap_or(true);
        if duplicate {
            packet.meta_mut().set_discard(true);
            num_duplicates += 1;
        }
    }
    let mut num_failed = 0;
    for packet in batch.iter_mut() {
        if !packet.meta().discard() && !verify_shred_cpu(packet, &slot_leaders, cache) {
            packet.meta_mut().set_discard(true);
            num_failed += 1;
        }
    }
    SigverifyOutput {
        index,
        batch,
        num_duplicates,
        num_failed,
    }
}

fn slot_leaders(slot: Slot, leader: Pubkey) -> SlotPubkeys {
    let mut slot_leaders = SlotPubkeys::default();
    slot_leaders.insert(slot, leader);
    slot_leaders.insert(Slot::MAX, Pubkey::default());
    slot_leaders
}

fn to_packet_batches(block: &BlockData) -> Vec<PacketBatch> {
    let packets: Vec<BytesPacket> = block
        .shreds
        .iter()
        .zip(&block.dropped)
        .filter(|(_, dropped)| !**dropped)
        .map(|(shred, _)| {
            let bytes = shred.payload().bytes.clone();
            let mut meta = Meta::default();
            meta.size = bytes.len();
            BytesPacket::new(bytes, meta)
        })
        .collect();
    packets
        .chunks(PACKETS_PER_BATCH)
        .map(|chunk| PacketBatch::from(chunk.to_vec()))
        .collect()
}

/// Per-slot stage times in microseconds.
#[derive(Clone, Copy, Default)]
struct SlotTimes {
    sigverify: u64,
    window: u64,
    insert: u64,
    replay: u64,
}

#[derive(Default)]
struct ReceiveStats {
    num_packets: usize,
    num_duplicates: usize,
    num_failed: usize,
    num_inserted_payloads: usize,
    num_duplicate_shreds: usize,
    shred_recovery_us: u64,
    write_batch_us: u64,
    num_recovered_data: usize,
    num_recovered_code: usize,
    num_fec_set_checks_passed: usize,
    num_fec_set_checks_failed: usize,
}

fn percentile(sorted: &[u64], percent: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = (sorted.len() * percent / 100).min(sorted.len() - 1);
    sorted[index] as f64 / 1e3
}

fn print_slot_stats(name: &str, times: &[SlotTimes], get: fn(&SlotTimes) -> u64) {
    let mut values: Vec<u64> = times.iter().map(get).collect();
    values.sort_unstable();
    println!(
        "SLOTSTATS:{name}: p50_ms={:.3} p90_ms={:.3} max_ms={:.3}",
        percentile(&values, 50),
        percentile(&values, 90),
        percentile(&values, 100),
    );
}

fn print_phase(name: &str, us: u64) {
    println!("BENCH_PHASE:{name}:{:.6}", micros(us));
}

/// Single-threaded SHA-256 work, untimed with respect to BENCH_RESULT: one
/// Merkle root pass over the measured shreds (the work stages A and B each do
/// once per shred), and plain 64-byte hashes.
fn print_sha256_side_measurements(blocks: &[BlockData]) {
    let start = Instant::now();
    let mut num_roots = 0usize;
    for block in blocks {
        for shred in &block.shreds {
            black_box(shred.merkle_root().unwrap());
            num_roots += 1;
        }
    }
    let merkle_elapsed = start.elapsed();
    const NUM_HASHES: usize = 1_000_000;
    let mut input = [0u8; 64];
    let start = Instant::now();
    for i in 0..NUM_HASHES {
        input[..8].copy_from_slice(&i.to_le_bytes());
        black_box(hashv(&[black_box(&input)]));
    }
    let hash_elapsed = start.elapsed();
    println!(
        "SHA256: merkle_roots={num_roots} merkle_root_us={:.3} hash64_ns={:.1}",
        merkle_elapsed.as_secs_f64() * 1e6 / num_roots.max(1) as f64,
        hash_elapsed.as_secs_f64() * 1e9 / NUM_HASHES as f64,
    );
    println!(
        "BENCH_PHASE:side_shred_merkle_1t:{:.6}",
        seconds(merkle_elapsed)
    );
    println!("BENCH_PHASE:side_sha256_64b_1m:{:.6}", seconds(hash_elapsed));
}

/// Stage D on an existing bank: the sequence of `process_single_slot`, with
/// the timings that `confirm_full_slot` does not report.
fn replay_bank(
    chain: &Chain,
    replay_verification_worker_pool: &ReplayVerificationWorkerPool,
    shred_version: u16,
    parent: &Arc<Bank>,
    bank: &BankWithScheduler,
    finish: Finish,
    stats: &mut ReplayStats,
) -> Result<(), BlockstoreProcessorError> {
    let blockstore = &chain.blockstore;
    let migration_status = chain.bank_forks.read().unwrap().migration_status();
    let slot = bank.slot();

    // Always Inactive for Alpenglow slots, which use double-Merkle block ids.
    let _ = check_chained_block_id(blockstore, bank, &migration_status);

    let mut timing = ConfirmationTiming::default();
    let mut progress = ConfirmationProgress::new(parent.last_blockhash());
    let start = Instant::now();
    confirm_slot(
        blockstore,
        bank,
        shred_version,
        replay_verification_worker_pool,
        &mut timing,
        &mut progress,
        false, // skip_verification
        None,
        None,
        false, // allow_dead_slots
        &migration_status,
    )?;
    stats.confirm_us += start.elapsed().as_micros() as u64;

    let start = Instant::now();
    stats
        .execute_timings
        .accumulate(&timing.batch_execute.totals);
    if !bank.is_complete() {
        return Err(BlockstoreProcessorError::InvalidBlock(
            BlockError::Incomplete,
        ));
    }
    if let Some((result, execute_timings)) = bank.wait_for_completed_scheduler() {
        stats.execute_timings.accumulate(&execute_timings);
        result?;
    }
    // confirm_full_slot passes &mut 0 here; keep the drained times instead.
    progress.wait_for_all_verification_results(
        &mut timing.poh_verify_elapsed,
        &mut timing.transaction_verify_elapsed,
    )?;
    stats.drain_us += start.elapsed().as_micros() as u64;

    let start = Instant::now();
    match finish {
        Finish::Verify => {
            let block_id = blockstore
                .get_block_id(slot, &migration_status)
                .unwrap()
                .expect("a full block has a block id");
            bank.set_block_id(Some(BlockId::from(block_id)));
            bank.freeze_and_verify_bank_hash()
                .map_err(|(expected, computed)| {
                    BlockstoreProcessorError::BankHashMismatch(slot, expected, computed)
                })?;
            blockstore.insert_bank_hash(slot, bank.hash(), false);
        }
        Finish::Discover => bank.freeze(),
    }
    stats.freeze_us += start.elapsed().as_micros() as u64;

    stats.fetch_us += timing.fetch_elapsed;
    stats.confirmation_us += timing.confirmation_elapsed;
    stats.replay_us += timing.replay_elapsed;
    stats.poh_verify_us += timing.poh_verify_elapsed;
    stats.sig_verify_us += timing.transaction_verify_elapsed;
    Ok(())
}

/// Stage D for a new slot: `Bank::new_from_parent` and `BankForks::insert`,
/// then `replay_bank`.
fn replay_slot(
    chain: &Chain,
    replay_verification_worker_pool: &ReplayVerificationWorkerPool,
    shred_version: u16,
    parent: &Arc<Bank>,
    slot: Slot,
    stats: &mut ReplayStats,
) -> Result<BankWithScheduler, BlockstoreProcessorError> {
    let start = Instant::now();
    let bank = chain.new_bank(parent, slot);
    stats.bank_new_us += start.elapsed().as_micros() as u64;
    replay_bank(
        chain,
        replay_verification_worker_pool,
        shred_version,
        parent,
        &bank,
        Finish::Verify,
        stats,
    )?;
    Ok(bank)
}

struct Measured {
    times: Vec<SlotTimes>,
    replay: ReplayStats,
    receive: ReceiveStats,
    block_id_mismatches: usize,
    bank_hash_mismatches: usize,
}

/// Runs stages A to D over every block and times the selected stages.
#[allow(clippy::too_many_arguments)]
fn run_measured(
    config: &Config,
    blocks: &[BlockData],
    chain: &Chain,
    replay_verification_worker_pool: &ReplayVerificationWorkerPool,
    shred_version: u16,
) -> Measured {
    let mode = config.mode;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    rng.set_stream(u64::MAX - 1);
    let deduper = Arc::new(Deduper::<2, [u8]>::new(&mut rng, DEDUPER_NUM_BITS));
    let cache = Arc::new(RwLock::new(LruCache::new(SIGVERIFY_LRU_CACHE_CAPACITY)));
    let pass_capacity = NonZeroUsize::new(SIGVERIFY_BATCHES_PER_PASS).unwrap();
    let sigverify_pool = ResultWorkerPool::spawn(
        WorkerPoolConfig {
            job_queue_capacity: pass_capacity,
            ..WorkerPoolConfig::new(
                "bbSigverify",
                NonZeroUsize::new(config.sigverify_threads).unwrap(),
            )
        },
        pass_capacity,
        move |_worker_id| {
            let deduper = deduper.clone();
            let cache = cache.clone();
            move |job: SigverifyJob| sigverify_batch(job, &deduper, &cache)
        },
    )
    .unwrap();
    let window_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.window_threads)
        .use_current_thread()
        .thread_name(|i| format!("bbWindow{i:02}"))
        .build()
        .unwrap();
    // Recovered shreds go to retransmit in production; here they are dropped.
    let retransmit_sender: RetransmitSender = Arc::new(
        WorkerPool::spawn(
            WorkerPoolConfig::new("bbRetransmit", NonZeroUsize::MIN),
            |_| |_: ShredBatch| {},
        )
        .unwrap(),
    );
    let blockstore = &chain.blockstore;
    let mut recovery_context =
        ShredRecoveryContext::new(retransmit_sender, chain.bank(0), shred_version);
    let mut pinnable_slice = blockstore.new_pinnable_slice();
    let mut write_batch = blockstore.get_write_batch();
    let num_duplicate_shreds = Cell::new(0usize);
    let handle_duplicate = |_| num_duplicate_shreds.set(num_duplicate_shreds.get() + 1);

    let mut packets: Vec<Vec<PacketBatch>> = if mode.runs_receive() {
        blocks.iter().map(to_packet_batches).collect()
    } else {
        Vec::new()
    };
    let mut measured = Measured {
        times: Vec::with_capacity(config.slots as usize),
        replay: ReplayStats::default(),
        receive: ReceiveStats::default(),
        block_id_mismatches: 0,
        bank_hash_mismatches: 0,
    };
    let mut warmup_replay = ReplayStats::default();
    let mut warmup_receive = ReceiveStats::default();
    let mut parent = chain.bank(0);

    for (i, block) in blocks.iter().enumerate() {
        let slot = block.slot;
        let is_measured = slot > config.warmup;
        let (replay_stats, receive_stats) = if is_measured {
            (&mut measured.replay, &mut measured.receive)
        } else {
            (&mut warmup_replay, &mut warmup_receive)
        };
        let mut times = SlotTimes::default();

        if mode.runs_receive() {
            let batches = std::mem::take(&mut packets[i]);
            let leaders = Arc::new(slot_leaders(slot, block.leader));
            receive_stats.num_packets += batches.iter().map(|batch| batch.len()).sum::<usize>();

            // Stage A.
            let start = Instant::now();
            let mut payloads = Vec::with_capacity(block.shreds.len());
            let mut batches = batches.into_iter().enumerate().peekable();
            while batches.peek().is_some() {
                let mut num_jobs = 0;
                for (index, batch) in batches.by_ref().take(SIGVERIFY_BATCHES_PER_PASS) {
                    sigverify_pool.send(SigverifyJob {
                        index,
                        batch,
                        slot_leaders: leaders.clone(),
                    });
                    num_jobs += 1;
                }
                let mut outputs: Vec<SigverifyOutput> = (0..num_jobs)
                    .map(|_| sigverify_pool.recv().expect("shred sigverify worker failed"))
                    .collect();
                // Keep the arrival order fixed so every run inserts shreds in
                // the same order.
                outputs.sort_unstable_by_key(|output| output.index);
                for output in outputs {
                    receive_stats.num_duplicates += output.num_duplicates;
                    receive_stats.num_failed += output.num_failed;
                    payloads.extend(
                        output
                            .batch
                            .iter()
                            .filter(|packet| !packet.meta().discard())
                            .map(|packet| Payload::from(packet.buffer().clone())),
                    );
                }
            }
            times.sigverify = start.elapsed().as_micros() as u64;

            // Stage B.
            let start = Instant::now();
            let shreds: Vec<_> = window_pool.install(|| {
                payloads
                    .into_par_iter()
                    .with_min_len(WINDOW_MIN_LEN)
                    .filter_map(|payload| {
                        let shred = Shred::new_from_serialized_shred(payload).ok()?;
                        let merkle_root = shred.merkle_root().ok();
                        Some((shred, false, BlockLocation::Original, merkle_root))
                    })
                    .collect()
            });
            times.window = start.elapsed().as_micros() as u64;
            receive_stats.num_inserted_payloads += shreds.len();

            // Stage C.
            let start = Instant::now();
            let mut metrics = BlockstoreInsertionMetrics::default();
            let mut shreds = shreds.into_iter();
            loop {
                let chunk: Vec<_> = shreds.by_ref().take(config.insert_chunk).collect();
                if chunk.is_empty() {
                    break;
                }
                blockstore
                    .insert_shreds_at_location_handle_duplicate(
                        chunk,
                        false, // is_trusted
                        &mut recovery_context,
                        &mut pinnable_slice,
                        &mut write_batch,
                        &handle_duplicate,
                        &mut metrics,
                    )
                    .unwrap();
            }
            times.insert = start.elapsed().as_micros() as u64;
            receive_stats.shred_recovery_us += metrics.shred_recovery_elapsed_us;
            receive_stats.write_batch_us += metrics.write_batch_elapsed_us;
            receive_stats.num_recovered_data += metrics.num_recovered_data_shreds;
            receive_stats.num_recovered_code += metrics.num_recovered_coding_shreds;
            receive_stats.num_fec_set_checks_passed += metrics.num_fec_set_checks_passed;
            receive_stats.num_fec_set_checks_failed += metrics.num_fec_set_checks_failed;
        }

        if mode.runs_replay() {
            // Stage D.
            let start = Instant::now();
            let bank = replay_slot(
                chain,
                replay_verification_worker_pool,
                shred_version,
                &parent,
                slot,
                replay_stats,
            )
            .unwrap_or_else(|err| panic!("slot {slot} failed to replay: {err}"));
            times.replay = start.elapsed().as_micros() as u64;
            if bank.block_id() != Some(BlockId::from(block.block_id)) {
                measured.block_id_mismatches += 1;
            }
            if bank.hash() != block.bank_hash {
                measured.bank_hash_mismatches += 1;
            }
            parent = bank.clone_without_scheduler();
            // Untimed: root slot - root_depth.
            if slot > config.root_depth {
                let root_bank = chain.set_root(slot - config.root_depth);
                recovery_context.maybe_update(root_bank);
            }
        } else {
            let migration_status = chain.bank_forks.read().unwrap().migration_status();
            let block_id = blockstore.get_block_id(slot, &migration_status).unwrap();
            if block_id != Some(block.block_id) {
                measured.block_id_mismatches += 1;
            }
        }
        if is_measured {
            measured.times.push(times);
        }
    }
    measured.receive.num_duplicate_shreds = num_duplicate_shreds.get();
    measured
}

/// Correctness checks for `--check`. Returns the failures.
#[allow(clippy::too_many_arguments)]
fn run_checks(
    config: &Config,
    world: &World,
    blocks: &[BlockData],
    measured: &Measured,
    chain: &Chain,
    generation: &Chain,
    replay_verification_worker_pool: &ReplayVerificationWorkerPool,
    shred_version: u16,
    accounts_dir: PathBuf,
) -> Vec<String> {
    let mut failures = Vec::new();
    let mut check = |ok: bool, what: String| {
        println!("CHECK:{}:{what}", if ok { "ok" } else { "FAILED" });
        if !ok {
            failures.push(what);
        }
    };
    let measured_blocks: Vec<&BlockData> = blocks
        .iter()
        .filter(|block| block.slot > config.warmup)
        .collect();
    let num_dropped_data: usize = measured_blocks
        .iter()
        .map(|block| block.num_dropped_data)
        .sum();
    let receive = &measured.receive;
    if config.mode.runs_receive() {
        check(
            receive.num_failed == 0 && receive.num_duplicates == 0,
            format!(
                "sigverify discards: failed={} duplicates={}",
                receive.num_failed, receive.num_duplicates
            ),
        );
        check(
            receive.num_recovered_data == num_dropped_data,
            format!(
                "recovered data shreds {} equal dropped data shreds {num_dropped_data}",
                receive.num_recovered_data
            ),
        );
        check(
            receive.num_fec_set_checks_failed == 0,
            format!(
                "FEC set check failures: {}",
                receive.num_fec_set_checks_failed
            ),
        );
    }
    check(
        measured.block_id_mismatches == 0,
        format!(
            "block ids equal the generated block ids, mismatches: {}",
            measured.block_id_mismatches
        ),
    );
    check(
        measured.bank_hash_mismatches == 0,
        format!(
            "bank hashes equal the generated bank hashes, mismatches: {}",
            measured.bank_hash_mismatches
        ),
    );

    // A flipped byte changes the Merkle root, so the leader signature fails.
    let first = &blocks[0];
    let leaders = slot_leaders(first.slot, first.leader);
    let cache = RwLock::new(LruCache::new(16));
    let mut bytes = first.shreds[0].payload().bytes.to_vec();
    let mut meta = Meta::default();
    meta.size = bytes.len();
    let good = BytesPacket::new(bytes.clone().into(), meta.clone());
    bytes[100] ^= 1;
    let bad = BytesPacket::new(bytes.into(), meta);
    check(
        verify_shred_cpu(&good, &leaders, &cache) && !verify_shred_cpu(&bad, &leaders, &cache),
        "a flipped shred byte fails sigverify".to_string(),
    );

    // Replay slot 1 through the pub process_single_slot on a new chain; it
    // must give the bank hash of the mirrored stage D.
    let check_chain = new_chain(
        &world.genesis_config,
        chain.blockstore.clone(),
        accounts_dir,
        config.handler_threads,
    );
    let migration_status = check_chain.bank_forks.read().unwrap().migration_status();
    let bank0 = check_chain.bank(0);
    let bank1 = check_chain.new_bank(&bank0, 1);
    let opts = ProcessOptions {
        run_verification: true,
        ..ProcessOptions::default()
    };
    let result = process_single_slot(
        &check_chain.blockstore,
        &bank1,
        shred_version,
        replay_verification_worker_pool,
        &opts,
        &mut ConfirmationProgress::new(bank0.last_blockhash()),
        None,
        None,
        &mut ExecuteTimings::default(),
        &migration_status,
    );
    check(
        result.is_ok() && bank1.hash() == first.bank_hash,
        format!(
            "process_single_slot gives the bank hash of slot 1: {result:?} {} {}",
            bank1.hash(),
            first.bank_hash
        ),
    );

    // The generation ledger holds blocks whose footer carries a placeholder
    // bank hash, so replaying slot 2 from it must fail the hash check.
    if blocks.len() > 1 {
        let placeholder_chain = Chain {
            bank_forks: check_chain.bank_forks.clone(),
            leader_schedule_cache: check_chain.leader_schedule_cache.clone(),
            blockstore: generation.blockstore.clone(),
        };
        let parent = bank1.clone_without_scheduler();
        let result = replay_slot(
            &placeholder_chain,
            replay_verification_worker_pool,
            shred_version,
            &parent,
            2,
            &mut ReplayStats::default(),
        );
        check(
            matches!(result, Err(BlockstoreProcessorError::BankHashMismatch(2, ..))),
            format!(
                "a wrong footer bank hash gives BankHashMismatch: {:?}",
                result.err()
            ),
        );
    }
    failures
}

fn fresh_dir(base: &Path, name: &str) -> PathBuf {
    let path = base.join(name);
    if path.exists() {
        fs::remove_dir_all(&path)
            .unwrap_or_else(|err| panic!("failed to remove {}: {err}", path.display()));
    }
    fs::create_dir_all(&path)
        .unwrap_or_else(|err| panic!("failed to create {}: {err}", path.display()));
    path
}

fn main() {
    agave_logger::setup();
    let config = parse_args();
    let mode_name = match config.mode {
        Mode::Full => "full",
        Mode::Shreds => "shreds",
        Mode::Replay => "replay",
    };
    let mix: Vec<String> = config
        .mix
        .iter()
        .map(|(class, weight)| format!("{}={weight}", class.name()))
        .collect();
    println!(
        "CONFIG: mode={mode_name} mix={} ({}) cu_target={} txs={} entry_txs={}-{} \
         memo_bytes={}-{} slots={} warmup={} voters={} payers={} drop_data={} drop_code={} \
         insert_chunk={} root_depth={} final_cert={} reward_cert={} seed={}",
        config.mix_name,
        mix.join(","),
        config.cu_target,
        config.txs,
        config.entry_txs.0,
        config.entry_txs.1,
        config.memo_bytes.0,
        config.memo_bytes.1,
        config.slots,
        config.warmup,
        config.voters,
        config.payers,
        config.drop_data,
        config.drop_code,
        config.insert_chunk,
        config.root_depth,
        config.final_cert,
        config.reward_cert,
        config.seed,
    );
    println!(
        "THREADS: handler={} verify={} sigverify={} window={}",
        config.handler_threads,
        config.verify_threads,
        config.sigverify_threads,
        config.window_threads
    );

    let temp_dir;
    let base_dir = match &config.ledger_dir {
        Some(dir) => {
            fs::create_dir_all(dir).unwrap();
            dir.clone()
        }
        None => {
            temp_dir = tempfile::tempdir().unwrap();
            temp_dir.path().to_path_buf()
        }
    };

    let setup_start = Instant::now();
    let start = Instant::now();
    let world = build_world(&config);
    let genesis_config = &world.genesis_config;
    let shred_version = solana_shred_version::compute_shred_version(&genesis_config.hash(), None);
    println!("SETUP:world:{:.3}", seconds(start.elapsed()));

    let start = Instant::now();
    let generation = new_chain(
        genesis_config,
        Arc::new(create_ledger(
            &fresh_dir(&base_dir, "ledger-generation"),
            genesis_config,
        )),
        fresh_dir(&base_dir, "accounts-generation"),
        config.handler_threads,
    );
    let final_blockstore = Arc::new(create_ledger(
        &fresh_dir(&base_dir, "ledger-final"),
        genesis_config,
    ));
    println!("SETUP:ledgers:{:.3}", seconds(start.elapsed()));

    let bank0 = generation.bank(0);
    let mut active_features: Vec<Pubkey> = bank0.feature_set.active().keys().copied().collect();
    active_features.sort_unstable();
    let feature_bytes: Vec<&[u8]> = active_features.iter().map(|id| id.as_ref()).collect();
    println!(
        "FEATURES: active={} inactive={} id={} stricter_shred_validation={}",
        active_features.len(),
        bank0.feature_set.inactive().len(),
        hashv(&feature_bytes),
        bank0
            .feature_set
            .is_active(&agave_feature_set::stricter_shred_validation::id()),
    );

    let replay_verification_worker_pool =
        ReplayVerificationWorkerPool::new(NonZeroUsize::new(config.verify_threads).unwrap());
    let generator = Generator {
        config: &config,
        world: &world,
    };
    // Calibrate on a bank for a slot after the generated range. It is never
    // inserted into the forks, and simulation stores nothing.
    let calibration = {
        let slot = config.warmup + config.slots + 2 * NUM_SLOTS_FOR_REWARD;
        let leader = generation
            .leader_schedule_cache
            .slot_leader_at(slot, Some(&bank0))
            .unwrap();
        generator.calibrate(&Bank::new_from_parent(bank0, leader, slot))
    };

    let start = Instant::now();
    let blocks = generate_blocks(
        &config,
        &world,
        &generation,
        &final_blockstore,
        &replay_verification_worker_pool,
        shred_version,
        &calibration,
    );
    println!("SETUP:generate:{:.3}", seconds(start.elapsed()));

    // MODE=replay replays the final ledger, which already holds every block.
    // The other modes insert the blocks into a new ledger first.
    let start = Instant::now();
    let measured_blockstore = if config.mode.runs_receive() {
        Arc::new(create_ledger(
            &fresh_dir(&base_dir, "ledger-measured"),
            genesis_config,
        ))
    } else {
        final_blockstore.clone()
    };
    let chain = new_chain(
        genesis_config,
        measured_blockstore,
        fresh_dir(&base_dir, "accounts-measured"),
        config.handler_threads,
    );
    println!("SETUP:measured_chain:{:.3}", seconds(start.elapsed()));
    println!("SETUP:total:{:.3}", seconds(setup_start.elapsed()));

    let start = Instant::now();
    let measured = run_measured(
        &config,
        &blocks,
        &chain,
        &replay_verification_worker_pool,
        shred_version,
    );
    println!("RUN:wall:{:.3}", seconds(start.elapsed()));

    let measured_blocks: Vec<&BlockData> = blocks
        .iter()
        .filter(|block| block.slot > config.warmup)
        .collect();
    let num_blocks = measured_blocks.len().max(1);
    let average = |get: fn(&BlockData) -> usize| {
        measured_blocks.iter().map(|block| get(block)).sum::<usize>() as f64 / num_blocks as f64
    };
    println!(
        "SHAPE: txs={:.1} sigs={:.1} entries={:.1} batches={:.1} fec_sets={:.1} shreds={:.1} \
         dropped_data={:.1} entry_bytes={:.0} estimated_cu={:.0} cost_units={:.0}",
        average(|block| block.num_txs),
        average(|block| block.num_sigs),
        average(|block| block.num_entries),
        average(|block| block.num_batches),
        average(|block| block.num_fec_sets),
        average(|block| block.shreds.len()),
        average(|block| block.num_dropped_data),
        average(|block| block.entry_bytes),
        measured_blocks
            .iter()
            .map(|block| block.estimated_cu)
            .sum::<u64>() as f64
            / num_blocks as f64,
        measured_blocks
            .iter()
            .map(|block| block.cost_units)
            .sum::<u64>() as f64
            / num_blocks as f64,
    );
    let receive = &measured.receive;
    if config.mode.runs_receive() {
        println!(
            "RECEIVE: packets={} sigverify_failed={} duplicates={} recovered_data={} \
             recovered_code={} fec_set_checks_passed={} fec_set_checks_failed={} \
             duplicate_shreds={}",
            receive.num_packets,
            receive.num_failed,
            receive.num_duplicates,
            receive.num_recovered_data,
            receive.num_recovered_code,
            receive.num_fec_set_checks_passed,
            receive.num_fec_set_checks_failed,
            receive.num_duplicate_shreds,
        );
    }

    let times = &measured.times;
    let total = |get: fn(&SlotTimes) -> u64| times.iter().map(get).sum::<u64>();
    let mut bench_result_us = 0;
    if config.mode.runs_receive() {
        print_slot_stats("shred_sigverify", times, |t| t.sigverify);
        print_slot_stats("window_merkle", times, |t| t.window);
        print_slot_stats("blockstore_insert", times, |t| t.insert);
        print_phase("shred_sigverify", total(|t| t.sigverify));
        print_phase("window_merkle", total(|t| t.window));
        print_phase("blockstore_insert", total(|t| t.insert));
        print_phase("insert_recovery", receive.shred_recovery_us);
        print_phase("insert_write", receive.write_batch_us);
        bench_result_us += total(|t| t.sigverify + t.window + t.insert);
    }
    if config.mode.runs_replay() {
        let replay = &measured.replay;
        let metrics = &replay.execute_timings.metrics;
        print_slot_stats("replay", times, |t| t.replay);
        print_phase("replay", total(|t| t.replay));
        print_phase("bank_new", replay.bank_new_us);
        print_phase("replay_fetch", replay.fetch_us);
        print_phase("replay_entries", replay.confirmation_us);
        print_phase("replay_schedule", replay.replay_us);
        print_phase("poh_verify_batches", replay.poh_verify_us);
        print_phase("sig_verify_batches", replay.sig_verify_us);
        print_phase(
            "footer_and_exec_wait",
            replay
                .confirm_us
                .saturating_sub(replay.confirmation_us + replay.fetch_us),
        );
        print_phase("replay_drain", replay.drain_us);
        print_phase("freeze", replay.freeze_us);
        // Thread time summed over the scheduler handler threads.
        print_phase("exec_load_thread", metrics[ExecuteTimingType::LoadUs].0);
        print_phase("exec_execute_thread", metrics[ExecuteTimingType::ExecuteUs].0);
        print_phase("exec_store_thread", metrics[ExecuteTimingType::StoreUs].0);
        bench_result_us += total(|t| t.replay);
    }
    print_sha256_side_measurements(&blocks[blocks.len() - measured_blocks.len()..]);
    println!(
        "FINAL_BANK_HASH:{}",
        blocks.last().map(|block| block.bank_hash).unwrap_or_default()
    );
    println!("BENCH_RESULT:{:.6}", micros(bench_result_us));

    if config.check {
        let failures = run_checks(
            &config,
            &world,
            &blocks,
            &measured,
            &chain,
            &generation,
            &replay_verification_worker_pool,
            shred_version,
            fresh_dir(&base_dir, "accounts-check"),
        );
        let ok = failures.is_empty() && bench_result_us > 0;
        println!("CHECK_RESULT:{}", if ok { "ok" } else { "FAILED" });
        if !ok {
            std::process::exit(1);
        }
    }
}
