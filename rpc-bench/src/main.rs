#![allow(clippy::arithmetic_side_effects)]
//! Benchmark of JSON RPC serving, including simulateTransaction.
//!
//! Builds a seeded world and a rooted history, serves them with
//! `JsonRpcService` on 127.0.0.1, and drives the service with a closed-loop
//! HTTP/1.1 client in the same process. BENCH_RESULT is the wall time of the
//! measured requests only.

mod client;
mod history;
mod requests;
mod server;
mod stats;
mod validate;
mod world;

#[cfg(target_os = "linux")]
use solana_clap_utils::input_parsers::parse_cpu_ranges;
use {
    clap::{App, Arg, ArgMatches, value_t_or_exit},
    requests::{Class, Mode},
    solana_account::{Account, ReadableAccount},
    solana_accounts_db::accounts_index::{AccountIndex, AccountSecondaryIndexes},
    solana_epoch_schedule::EpochSchedule,
    solana_fee_calculator::FeeRateGovernor,
    solana_keypair::Keypair,
    solana_ledger::blockstore::Blockstore,
    solana_rent::Rent,
    solana_rpc::rpc::{JsonRpcConfig, MAX_REQUEST_BODY_SIZE},
    solana_runtime::genesis_utils::create_genesis_config_with_leader,
    solana_signer::Signer,
    std::{collections::HashSet, fs, path::PathBuf, process::exit, sync::Arc, time::Instant},
};

#[cfg(not(any(target_env = "msvc", target_os = "freebsd")))]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

const LAMPORTS_PER_SOL: u64 = 1_000_000_000;

/// The CPUs this process may use, split into the server and client sets.
struct CpuSplit {
    all: usize,
    #[cfg(target_os = "linux")]
    server: Vec<agave_cpu_utils::CpuId>,
    #[cfg(target_os = "linux")]
    client: Vec<agave_cpu_utils::CpuId>,
}

impl CpuSplit {
    /// Gives the client whole physical cores from the end of the allowed
    /// set, so that it does not share a core with the server through SMT.
    #[cfg(target_os = "linux")]
    fn new(client_cpus: usize, pin: bool) -> Self {
        let allowed = agave_cpu_utils::cpu_affinity(None).unwrap();
        let all = allowed.len();
        let unpinned = Self {
            all,
            server: vec![],
            client: vec![],
        };
        if !pin || client_cpus >= all {
            return unpinned;
        }
        let ids: Vec<usize> = allowed.iter().map(|cpu| **cpu).collect();
        let client = take_cores(&ids, client_cpus, |cpu| {
            fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ))
            .ok()
            .and_then(|list| parse_cpu_ranges(list.trim()).ok())
            .unwrap_or_else(|| vec![cpu])
        });
        let (client, server): (Vec<_>, Vec<_>) =
            allowed.into_iter().partition(|cpu| client.contains(cpu));
        if server.is_empty() {
            return unpinned;
        }
        Self {
            all,
            server,
            client,
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn new(_client_cpus: usize, _pin: bool) -> Self {
        Self {
            all: std::thread::available_parallelism().map_or(1, |n| n.get()),
        }
    }

    fn server_cpus(&self) -> usize {
        #[cfg(target_os = "linux")]
        if !self.server.is_empty() {
            return self.server.len();
        }
        self.all
    }

    /// The number of client CPUs, which is `requested` rounded up to whole
    /// cores when the CPUs are split.
    fn client_cpus(&self, requested: usize) -> usize {
        #[cfg(target_os = "linux")]
        if !self.client.is_empty() {
            return self.client.len();
        }
        requested
    }

    fn describe(&self) -> String {
        #[cfg(target_os = "linux")]
        if !self.server.is_empty() {
            let list = |cpus: &[agave_cpu_utils::CpuId]| {
                cpus.iter()
                    .map(|cpu| cpu.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            return format!(
                "server_cpus={} client_cpus={}",
                list(&self.server),
                list(&self.client)
            );
        }
        "not pinned".to_string()
    }

    /// Restricts the calling thread, and the threads it starts later, to
    /// the server or the client set.
    fn pin(&self, server: bool) {
        #[cfg(target_os = "linux")]
        if !self.server.is_empty() {
            let cpus = if server { &self.server } else { &self.client };
            agave_cpu_utils::set_cpu_affinity(None, cpus.iter().copied()).unwrap();
        }
        #[cfg(not(target_os = "linux"))]
        let _ = server;
    }
}

/// Takes whole cores from the end of `allowed` until at least `count` CPUs
/// are taken. `siblings` returns the CPUs of the core that a CPU belongs to.
/// Their numbering depends on the machine: cpu0 shares a core with cpu1 on
/// some, and with cpu4 on an 8-vCPU AWS Intel instance.
#[cfg(target_os = "linux")]
fn take_cores(
    allowed: &[usize],
    count: usize,
    siblings: impl Fn(usize) -> Vec<usize>,
) -> Vec<usize> {
    let mut taken = vec![];
    for &cpu in allowed.iter().rev() {
        if taken.len() >= count {
            break;
        }
        if !taken.contains(&cpu) {
            let core = siblings(cpu);
            taken.extend(allowed.iter().filter(|other| core.contains(other)));
        }
    }
    taken
}

/// The value of an optional argument, or `default` if it is not given.
/// Exits if the value does not parse.
fn value_or<T: std::str::FromStr>(matches: &ArgMatches, name: &str, default: T) -> T {
    if matches.is_present(name) {
        value_t_or_exit!(matches, name, T)
    } else {
        default
    }
}

fn parse_account_indexes(value: &str) -> AccountSecondaryIndexes {
    let indexes = if value == "none" {
        HashSet::new()
    } else {
        value
            .split(',')
            .map(|name| match name {
                "program-id" => AccountIndex::ProgramId,
                "spl-token-owner" => AccountIndex::SplTokenOwner,
                "spl-token-mint" => AccountIndex::SplTokenMint,
                _ => {
                    eprintln!("unknown account index: {name}");
                    exit(2);
                }
            })
            .collect()
    };
    AccountSecondaryIndexes {
        keys: None,
        indexes,
    }
}

fn main() {
    agave_logger::setup_with_default("error");
    let matches = App::new("agave-rpc-bench")
        .about(
            "Serves a synthetic rooted ledger with JsonRpcService on 127.0.0.1 and measures the \
             time to answer a fixed number of requests",
        )
        .arg(
            Arg::with_name("mode")
                .long("mode")
                .takes_value(true)
                .possible_values(&["mixed", "light", "medium", "heavy", "exec"])
                .default_value("mixed")
                .help("Request mix: all classes, or one class; exec is simulateTransaction"),
        )
        .arg(
            Arg::with_name("requests")
                .long("requests")
                .takes_value(true)
                .help(
                    "Number of measured requests [default: mixed 400000, light 5000000, medium \
                     60000, heavy 3000, exec 300000]",
                ),
        )
        .arg(
            Arg::with_name("seed")
                .long("seed")
                .takes_value(true)
                .default_value("42")
                .help("Seed of the world, the history and the request order"),
        )
        .arg(
            Arg::with_name("concurrency")
                .long("concurrency")
                .takes_value(true)
                .help(
                    "Number of connections, each with one request in flight [default: 32, light \
                     128]",
                ),
        )
        .arg(
            Arg::with_name("scale")
                .long("scale")
                .takes_value(true)
                .possible_values(&["full", "small"])
                .default_value("full")
                .help("World size: full is about 596K accounts, small is 1/20 of it"),
        )
        .arg(
            Arg::with_name("history_slots")
                .long("history-slots")
                .takes_value(true)
                .help("Slots with transactions after the warm-up slots [default: 64, small 4]"),
        )
        .arg(
            Arg::with_name("transactions_per_slot")
                .long("transactions-per-slot")
                .takes_value(true)
                .help("Transactions per history slot [default: 600, small 100]"),
        )
        .arg(
            Arg::with_name("account_indexes")
                .long("account-indexes")
                .takes_value(true)
                .default_value("program-id,spl-token-owner,spl-token-mint")
                .help("Comma-separated secondary account indexes, or none"),
        )
        .arg(
            Arg::with_name("rpc_threads")
                .long("rpc-threads")
                .takes_value(true)
                .help("RPC worker threads [default: number of server CPUs]"),
        )
        .arg(
            Arg::with_name("rpc_blocking_threads")
                .long("rpc-blocking-threads")
                .takes_value(true)
                .help("RPC blocking threads [default: max(1, CPUs / 4), as the validator]"),
        )
        .arg(
            Arg::with_name("client_cpus")
                .long("client-cpus")
                .takes_value(true)
                .help(
                    "CPUs reserved for the client, rounded up to whole cores [default: max(2, \
                     CPUs / 4), light max(2, CPUs / 2)]",
                ),
        )
        .arg(
            Arg::with_name("no_pin")
                .long("no-pin")
                .help("Do not split the CPUs between the server and the client"),
        )
        .arg(
            Arg::with_name("data_dir")
                .long("data-dir")
                .takes_value(true)
                .help(
                    "Directory for the accounts and the ledger; a run-<pid> directory inside it \
                     is deleted at exit [default: $HOME/rpc-bench-data]",
                ),
        )
        .arg(
            Arg::with_name("validate")
                .long("validate")
                .help("Check every request type against the generated data instead of measuring"),
        )
        .get_matches();

    let setup_start = Instant::now();
    let mode = match matches.value_of("mode").unwrap() {
        "mixed" => Mode::Mixed,
        "light" => Mode::Class(Class::Light),
        "medium" => Mode::Class(Class::Medium),
        "heavy" => Mode::Class(Class::Heavy),
        _ => Mode::Class(Class::Exec),
    };
    let requests = value_or(&matches, "requests", mode.default_requests());
    let seed = value_t_or_exit!(matches, "seed", u64);
    let concurrency = value_or(&matches, "concurrency", mode.default_concurrency());
    let small = matches.value_of("scale") == Some("small");
    let scale = if small { 20 } else { 1 };
    let history_slots = value_or(&matches, "history_slots", if small { 4 } else { 64 });
    let transactions_per_slot = value_or(
        &matches,
        "transactions_per_slot",
        if small { 100 } else { 600 },
    );
    let account_indexes = parse_account_indexes(matches.value_of("account_indexes").unwrap());
    let validate = matches.is_present("validate");

    let nproc = CpuSplit::new(0, false).all;
    let client_cpus = value_or(&matches, "client_cpus", mode.default_client_cpus(nproc));
    let cpus = CpuSplit::new(client_cpus, !matches.is_present("no_pin"));
    let client_cpus = cpus.client_cpus(client_cpus);
    let rpc_threads = value_or(&matches, "rpc_threads", cpus.server_cpus());
    let rpc_blocking_threads = value_or(&matches, "rpc_blocking_threads", (nproc / 4).max(1));

    let data_dir = matches.value_of("data_dir").map_or_else(
        || {
            std::env::var_os("HOME")
                .map_or_else(|| PathBuf::from("."), PathBuf::from)
                .join("rpc-bench-data")
        },
        PathBuf::from,
    );
    let run_dir = data_dir.join(format!("run-{}", std::process::id()));
    fs::create_dir_all(&run_dir).unwrap();

    println!(
        "mode={} requests={requests} seed={seed} concurrency={concurrency} scale={} \
         history_slots={history_slots} transactions_per_slot={transactions_per_slot} \
         account_indexes={} rpc_threads={rpc_threads} rpc_blocking_threads={rpc_blocking_threads} \
         cpus={nproc} {}",
        matches.value_of("mode").unwrap(),
        matches.value_of("scale").unwrap(),
        matches.value_of("account_indexes").unwrap(),
        cpus.describe(),
    );

    let mut rng = world::new_rng(seed);
    let mut world = world::World::new(&mut rng, &world::Sizes::new(scale));
    let identity = Arc::new(Keypair::new_from_array([1; 32]));
    let mut genesis = create_genesis_config_with_leader(
        1_000_000 * LAMPORTS_PER_SOL,
        &identity.pubkey(),
        1_000_000 * LAMPORTS_PER_SOL,
    );
    let genesis_config = &mut genesis.genesis_config;
    // The test genesis has no rent, no fees and an epoch schedule with
    // warm-up epochs. Use the mainnet values, and keep the whole history in
    // epoch 0.
    genesis_config.rent = Rent::default();
    genesis_config.fee_rate_governor = FeeRateGovernor::default();
    genesis_config.epoch_schedule = EpochSchedule::without_warmup();
    // A fixed creation time keeps block times, and so the response bytes,
    // the same on every run.
    genesis_config.creation_time = 1_700_000_000;
    let rent = genesis_config.rent.clone();
    let programs: Vec<_> = solana_program_binaries::spl_programs(&rent)
        .into_iter()
        .chain(solana_program_binaries::core_bpf_programs(
            &rent,
            |feature| genesis_config.accounts.contains_key(feature),
        ))
        .collect();
    for (address, account) in programs {
        genesis_config
            .accounts
            .entry(address)
            .or_insert_with(|| Account::from(account));
    }
    let genesis_hash = genesis.genesis_config.hash();

    // The world is too large for a genesis config, whose serialized size is
    // limited, so it is stored into the root bank instead. Stores update the
    // secondary indexes like any other write.
    let bank = history::new_bank(&genesis, &run_dir.join("accounts"), &account_indexes);
    let accounts = world.accounts(&mut rng, &rent);
    let lamports: u64 = accounts.iter().map(|(_, account)| account.lamports()).sum();
    for chunk in accounts.chunks(10_000) {
        bank.store_accounts((bank.slot(), chunk), None);
    }
    bank.set_capitalization_for_tests(bank.capitalization() + lamports);
    drop(accounts);
    let blockstore = Arc::new(Blockstore::open(&run_dir.join("ledger")).unwrap());
    let history = history::build(
        &mut rng,
        &world,
        &genesis,
        bank,
        &blockstore,
        history_slots,
        transactions_per_slot,
        nproc,
    );
    println!(
        "world accounts={} history_slots={} transactions={} executed={} failed={} tip_slot={}",
        world.num_accounts(),
        history.transaction_slots.len(),
        history.transactions.len(),
        history.executed,
        history.failed,
        history.tip.slot(),
    );

    let pools = Arc::new(requests::build_pools(&mut rng, &world, &history, scale));
    // The warmup fills the program cache, the read cache and the RocksDB
    // caches. A tenth of the run is enough; a fixed minimum would make the
    // heavy warmup longer than the heavy run.
    let warmup_requests = (requests / 10).clamp(100, 20_000);
    let warmup = Arc::new(requests::sequence(
        &mut world::new_rng(seed + 1),
        mode,
        warmup_requests,
        &pools,
    ));
    let measured = Arc::new(requests::sequence(&mut rng, mode, requests, &pools));

    let rpc_config = JsonRpcConfig {
        enable_rpc_transaction_history: true,
        enable_extended_tx_metadata_storage: true,
        max_multiple_accounts: Some(solana_rpc_client_api::request::MAX_MULTIPLE_ACCOUNTS),
        account_indexes: account_indexes.clone(),
        rpc_threads,
        rpc_blocking_threads,
        full_api: true,
        max_request_body_size: Some(MAX_REQUEST_BODY_SIZE),
        disable_health_check: true,
        ..JsonRpcConfig::default()
    };
    // Threads inherit the CPU set of the thread that starts them, so the
    // service runtime starts on the server CPUs and the client on the rest.
    cpus.pin(true);
    let server = server::Server::start(
        identity,
        &history,
        blockstore.clone(),
        &run_dir.join("ledger"),
        genesis_hash,
        rpc_config,
    );
    let workers = stats::thread_cpu();
    cpus.pin(false);
    let runtime = client::new_runtime(client_cpus);
    println!("rpc_addr={}", server.addr);
    println!(
        "BENCH_PHASE:setup:{:.6}",
        setup_start.elapsed().as_secs_f64()
    );

    let result = if validate {
        validate::run(
            &mut rng,
            &runtime,
            server.addr,
            &world,
            &history,
            &blockstore,
            transactions_per_slot,
        )
        .map(|()| println!("validate: all checks passed"))
    } else {
        measure(
            &runtime,
            &server,
            &pools,
            &warmup,
            &measured,
            concurrency,
            client_cpus,
            &workers,
        )
    };

    server.stop();
    fs::remove_dir_all(&run_dir).unwrap();
    if let Err(err) = result {
        println!("error: {err}");
        exit(1);
    }
    // Exit without dropping the banks and the blockstore, which is slow and
    // not part of the benchmark.
    exit(0);
}

fn measure(
    runtime: &tokio::runtime::Runtime,
    server: &server::Server,
    pools: &Arc<Vec<Vec<hyper::body::Bytes>>>,
    warmup: &Arc<Vec<(usize, usize)>>,
    measured: &Arc<Vec<(usize, usize)>>,
    concurrency: usize,
    client_cpus: usize,
    workers: &std::collections::BTreeMap<u32, (String, u64)>,
) -> Result<(), String> {
    let outcome = client::run(
        runtime,
        server.addr,
        pools.clone(),
        warmup.clone(),
        concurrency,
    )?;
    println!("BENCH_PHASE:warmup:{:.6}", outcome.elapsed.as_secs_f64());
    let start = stats::thread_cpu();
    let outcome = client::run(
        runtime,
        server.addr,
        pools.clone(),
        measured.clone(),
        concurrency,
    )?;
    let end = stats::thread_cpu();
    stats::report(&outcome.records, outcome.elapsed);
    stats::report_cpu(workers, &start, &end, outcome.elapsed, client_cpus);
    println!("BENCH_RESULT:{:.6}", outcome.elapsed.as_secs_f64());
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    fn client_of_8(count: usize, siblings: impl Fn(usize) -> Vec<usize>) -> Vec<usize> {
        let allowed: Vec<usize> = (0..8).collect();
        let mut client = take_cores(&allowed, count, siblings);
        client.sort_unstable();
        client
    }

    #[test]
    fn test_take_cores() {
        // One CPU per core, as on Graviton and c8a.
        assert_eq!(client_of_8(2, |cpu| vec![cpu]), [6, 7]);
        // Siblings N and N + 4, as on an 8-vCPU AWS Intel instance.
        let intel = |cpu: usize| vec![cpu % 4, cpu % 4 + 4];
        assert_eq!(client_of_8(2, intel), [3, 7]);
        assert_eq!(client_of_8(3, intel), [2, 3, 6, 7]);
        // Adjacent siblings.
        assert_eq!(client_of_8(2, |cpu| vec![cpu & !1, cpu | 1]), [6, 7]);
        // Siblings outside the allowed set are not taken.
        let allowed = [0, 1, 2, 3];
        assert_eq!(take_cores(&allowed, 2, intel), [3, 2]);
    }
}
