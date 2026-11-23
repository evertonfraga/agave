//! Measures the CPU time one validator spends validating Alpenglow consensus traffic.
//!
//! The benchmark replays a fixed number of slots of deterministic consensus traffic, as one
//! node of a synthetic cluster receives it, through three timed phases:
//! - sigverify: BLS vote and certificate verification, one `SigVerifier` pass at a time, with
//!   the passes cut from the arrival-ordered traffic by a fixed time window;
//! - cert_build: certificate building from the verified votes after every pass, as the
//!   consensus pool does it, so that the sigverifier skips the certificates the node has built;
//! - footer: validation of the block footer certificates, as replay does it.
//!
//! BENCH_RESULT is the sum of the three phases over all rounds. Each round replays the same
//! traffic with a new sigverifier.

#![allow(clippy::arithmetic_side_effects)]

mod traffic;

use {
    crate::traffic::{Cluster, Rng, Traffic, TrafficConfig},
    agave_bls_sigverify::{
        bls_sigverifier::{
            NUM_SLOTS_FOR_VERIFY, SigVerifier, SigVerifierChannels, SigVerifierContext,
        },
        generated_cert_types::GeneratedCertTypes,
        sig_verified_messages::{SigVerifiedBatch, VoteAggregate},
    },
    agave_votor::aggregate_accumulator::AggregateAccumulator,
    agave_votor_messages::{
        certificate::{Certificate, CertificateType},
        consensus_message::{Block, BlockId, VoteMessage},
        stake::NonZeroStake,
        unverified_vote_message::UnverifiedCertificate,
        vote::Vote,
    },
    agave_votor_transport::endpoint::stub_ban_channel_for_tests,
    clap::{App, Arg, crate_description, crate_name, value_t_or_exit},
    crossbeam_channel::unbounded,
    solana_pubkey::Pubkey,
    solana_runtime::{
        bank::MAX_ALPENGLOW_VOTE_ACCOUNTS,
        validated_block_finalization::ValidatedBlockFinalizationCert,
        validated_reward_certificate::ValidatedRewardCert,
    },
    solana_streamer::evicting_sender::EvictingSender,
    std::{
        collections::{HashMap, HashSet},
        num::NonZeroUsize,
        sync::{Arc, RwLock},
        time::{Duration, Instant},
    },
};

const REPAIR_CHANNEL_CAPACITY: usize = 4_096;
const BAN_CHANNEL_CAPACITY: usize = 4_096;
/// With a pass window below the smallest peer latency, a certificate copy that does not race
/// the node's own threshold crossing always arrives after the node built the certificate.
const MIN_LATENCY_US: u64 = 5_000;

struct Config {
    validators: usize,
    rounds: usize,
    threads: usize,
    seed: u64,
    check: bool,
    traffic: TrafficConfig,
}

impl Config {
    fn from_args() -> Self {
        let matches = App::new(crate_name!())
            .about(crate_description!())
            .arg(
                Arg::with_name("validators")
                    .long("validators")
                    .value_name("N")
                    .takes_value(true)
                    .default_value("700")
                    .help("Number of staked validators"),
            )
            .arg(
                Arg::with_name("slots")
                    .long("slots")
                    .value_name("N")
                    .takes_value(true)
                    .default_value("32")
                    .help("Number of distinct slots of traffic"),
            )
            .arg(
                Arg::with_name("rounds")
                    .long("rounds")
                    .value_name("N")
                    .takes_value(true)
                    .default_value("3")
                    .help("Number of times the traffic is replayed"),
            )
            .arg(
                Arg::with_name("pass_window_us")
                    .long("pass-window-us")
                    .value_name("MICROSECONDS")
                    .takes_value(true)
                    .default_value("1500")
                    .help(
                        "Datagrams that arrive within this window of the first pending one are \
                         verified in one pass. \"inf\" makes every pass take the 5000 datagram \
                         cap. 0 makes every pass take one datagram.",
                    ),
            )
            .arg(
                Arg::with_name("cert_copies")
                    .long("cert-copies")
                    .value_name("N")
                    .takes_value(true)
                    .default_value("all")
                    .help("Number of peers that send a copy of each certificate"),
            )
            .arg(
                Arg::with_name("cert_race")
                    .long("cert-race")
                    .value_name("FRACTION")
                    .takes_value(true)
                    .default_value("0.25")
                    .help(
                        "Fraction of certificates whose first copy arrives before the node builds \
                         the certificate itself, so that the copy is verified",
                    ),
            )
            .arg(
                Arg::with_name("bad_votes_per_slot")
                    .long("bad-votes-per-slot")
                    .value_name("N")
                    .takes_value(true)
                    .default_value("0")
                    .help("Number of finalize votes per slot with an invalid signature"),
            )
            .arg(
                Arg::with_name("threads")
                    .long("threads")
                    .value_name("N")
                    .takes_value(true)
                    .default_value("0")
                    .help("Sigverifier threads, 0 for the available parallelism"),
            )
            .arg(
                Arg::with_name("seed")
                    .long("seed")
                    .value_name("N")
                    .takes_value(true)
                    .default_value("42")
                    .help("Seed for keys and traffic"),
            )
            .arg(
                Arg::with_name("check")
                    .long("check")
                    .help("Check the results of every round and print the certificates"),
            )
            .get_matches();

        let validators = value_t_or_exit!(matches, "validators", usize);
        assert!(
            (2..=MAX_ALPENGLOW_VOTE_ACCOUNTS).contains(&validators),
            "--validators must be in 2..={MAX_ALPENGLOW_VOTE_ACCOUNTS}"
        );
        let slots = value_t_or_exit!(matches, "slots", u64);
        assert!(slots > 0, "--slots must be positive");
        let pass_window_us = match matches.value_of("pass_window_us").unwrap() {
            "inf" => None,
            _ => Some(value_t_or_exit!(matches, "pass_window_us", u64)),
        };
        let cert_copies = match matches.value_of("cert_copies").unwrap() {
            "all" => validators - 1,
            _ => value_t_or_exit!(matches, "cert_copies", usize),
        };
        let cert_race = value_t_or_exit!(matches, "cert_race", f64);
        assert!(
            (0.0..=1.0).contains(&cert_race),
            "--cert-race must be in 0..=1"
        );
        let threads = match value_t_or_exit!(matches, "threads", usize) {
            0 => std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
            threads => threads,
        };
        Self {
            validators,
            rounds: value_t_or_exit!(matches, "rounds", usize),
            threads,
            seed: value_t_or_exit!(matches, "seed", u64),
            check: matches.is_present("check"),
            traffic: TrafficConfig {
                slots,
                pass_window_us,
                cert_copies,
                cert_race,
                bad_votes_per_slot: value_t_or_exit!(matches, "bad_votes_per_slot", usize),
            },
        }
    }
}

/// Builds base2 certificates from the verified votes after every pass, in the order the
/// consensus pool tries them, and records them in `GeneratedCertTypes` so that the sigverifier
/// skips copies of them. Certificates verified by the sigverifier complete their type without
/// being built.
struct CertBuilder<'a> {
    own_votes: &'a HashMap<Vote, VoteMessage>,
    max_validators: usize,
    total_stake: NonZeroStake,
    generated_cert_types: Arc<GeneratedCertTypes>,
    accumulators: HashMap<Vote, AggregateAccumulator>,
    completed: HashSet<CertificateType>,
    built: Vec<Certificate>,
    received: Vec<Certificate>,
    /// Every verified (vote, rank). Only recorded with `--check`.
    verified_votes: Option<Vec<(Vote, usize)>>,
}

impl CertBuilder<'_> {
    fn process(&mut self, batches: Vec<SigVerifiedBatch>) {
        // Certificates go first, so that the result does not depend on the order in which the
        // sigverifier's threads sent the batches.
        let mut vote_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            match batch {
                SigVerifiedBatch::Certificates(certs) => {
                    for cert in certs {
                        self.completed.insert(cert.cert_type);
                        self.received.push(cert);
                    }
                }
                SigVerifiedBatch::Votes(aggregates) => vote_batches.push(aggregates),
            }
        }
        for aggregate in vote_batches.iter().flatten() {
            self.add_aggregate(aggregate);
        }
    }

    fn add_aggregate(&mut self, aggregate: &VoteAggregate) {
        let vote = *aggregate.vote();
        if let Some(verified_votes) = &mut self.verified_votes {
            verified_votes.extend(aggregate.ranks().iter_ones().map(|rank| (vote, rank)));
        }
        let accumulator = self.accumulators.entry(vote).or_insert_with(|| {
            let mut accumulator = AggregateAccumulator::new(self.max_validators);
            if let Some(own_vote) = self.own_votes.get(&vote) {
                accumulator.add_own_vote_message(own_vote).unwrap();
            }
            accumulator
        });
        accumulator.add_aggregate(aggregate).unwrap();
        // The base2 part of `VotePool::try_produce_cert` in agave-votor.
        let (first, second) = match vote {
            Vote::Notarize(notarize) => (
                CertificateType::FinalizeFast(notarize.block),
                Some(CertificateType::Notarize(notarize.block)),
            ),
            Vote::Finalize(finalize) => (CertificateType::Finalize(finalize.slot), None),
            Vote::Skip(skip) => (CertificateType::Skip(skip.slot), None),
            Vote::NotarizeFallback(_) | Vote::SkipFallback(_) | Vote::Genesis(_) => return,
        };
        for cert_type in std::iter::once(first).chain(second) {
            if self.completed.contains(&cert_type) {
                break;
            }
            if let Some(cert) = accumulator
                .try_build_base2_cert(cert_type, self.total_stake)
                .unwrap()
            {
                self.completed.insert(cert_type);
                self.generated_cert_types.insert_cert(cert_type);
                self.built.push(cert);
                break;
            }
        }
    }
}

#[derive(Default)]
struct Timings {
    sigverify: Duration,
    cert_build: Duration,
    footer: Duration,
}

struct RoundOutput {
    built: Vec<Certificate>,
    received: Vec<Certificate>,
    completed: HashSet<CertificateType>,
    verified_votes: Option<Vec<(Vote, usize)>>,
    banned: Vec<Pubkey>,
    valid_footers: usize,
}

fn run_round(
    cluster: &Cluster,
    traffic: &Traffic,
    config: &Config,
    timings: &mut Timings,
) -> RoundOutput {
    let (channel_to_pool, pool_receiver) = unbounded();
    let (channel_to_repair, repair_receiver) = EvictingSender::new_bounded(REPAIR_CHANNEL_CAPACITY);
    let (channel_to_reward, reward_receiver) = unbounded();
    let (channel_to_metrics, metrics_receiver) = unbounded();
    // The benchmark passes the inputs to the sigverifier directly, so these stay empty.
    let (_packet_sender, packet_receiver) = unbounded();
    let (_certificate_sender, certificate_receiver) = unbounded();
    let (ban_sender, mut ban_receiver) = stub_ban_channel_for_tests(BAN_CHANNEL_CAPACITY);
    let generated_cert_types = Arc::new(GeneratedCertTypes::default());
    let context = SigVerifierContext {
        ban_sender,
        sharable_banks: cluster.bank_forks.read().unwrap().sharable_banks(),
        highest_parent_ready: Arc::new(RwLock::new((
            NUM_SLOTS_FOR_VERIFY,
            Block {
                slot: NUM_SLOTS_FOR_VERIFY - 1,
                block_id: BlockId::default(),
            },
        ))),
        cluster_info: cluster.cluster_info.clone(),
        leader_schedule: cluster.leader_schedule.clone(),
        num_threads: config.threads,
        generated_cert_types: generated_cert_types.clone(),
    };
    let channels = SigVerifierChannels::new(
        packet_receiver,
        certificate_receiver,
        channel_to_repair,
        channel_to_reward,
        channel_to_pool,
        channel_to_metrics,
    );
    let mut verifier = SigVerifier::new(context, channels);
    let mut builder = CertBuilder {
        own_votes: &traffic.own_votes,
        max_validators: cluster.rank_map.len(),
        total_stake: cluster.rank_map.total_stake(),
        generated_cert_types,
        accumulators: HashMap::new(),
        completed: HashSet::new(),
        built: Vec::new(),
        received: Vec::new(),
        verified_votes: config.check.then(Vec::new),
    };
    let mut banned = Vec::new();

    for pass in &traffic.passes {
        let certificates = pass.certificates.clone();
        let start = Instant::now();
        verifier
            .verify_and_send_for_bench(&traffic.datagrams[pass.datagrams.clone()], certificates);
        timings.sigverify += start.elapsed();

        let start = Instant::now();
        builder.process(pool_receiver.try_iter().collect());
        timings.cert_build += start.elapsed();

        // Other consumers of the sigverifier's output, which are not measured.
        repair_receiver.try_iter().for_each(drop);
        reward_receiver.try_iter().for_each(drop);
        metrics_receiver.try_iter().for_each(drop);
        while let Ok(ban) = ban_receiver.try_recv() {
            banned.push(ban.peer);
        }
    }

    // Replay validates each footer, which consumes the final certificate.
    let final_certs: Vec<_> = traffic
        .footers
        .iter()
        .map(|footer| footer.final_cert.clone())
        .collect();
    let start = Instant::now();
    let mut valid_footers = 0;
    for (footer, final_cert) in traffic.footers.iter().zip(final_certs) {
        let reward_cert = ValidatedRewardCert::try_new(
            &footer.bank,
            cluster.shred_version,
            &footer.skip_reward_cert,
            &footer.notar_reward_cert,
        );
        let finalization_cert = ValidatedBlockFinalizationCert::try_from_footer(
            final_cert,
            &footer.bank,
            cluster.shred_version,
        );
        if matches!(reward_cert, Ok(Some(_))) && finalization_cert.is_ok() {
            valid_footers += 1;
        }
    }
    timings.footer += start.elapsed();

    RoundOutput {
        built: builder.built,
        received: builder.received,
        completed: builder.completed,
        verified_votes: builder.verified_votes,
        banned,
        valid_footers,
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Checks one round's results against the generated traffic. Panics on a mismatch.
fn check(cluster: &Cluster, traffic: &Traffic, config: &Config, output: &RoundOutput) {
    let verified_votes = output.verified_votes.as_ref().unwrap();
    let unique_votes: HashSet<_> = verified_votes.iter().copied().collect();
    assert_eq!(
        unique_votes.len(),
        verified_votes.len(),
        "duplicate verified votes"
    );
    assert!(
        unique_votes == traffic.valid_votes,
        "verified votes differ from the valid votes: {} verified, {} valid",
        unique_votes.len(),
        traffic.valid_votes.len()
    );

    let banned: HashSet<_> = output.banned.iter().copied().collect();
    assert_eq!(output.banned.len(), traffic.bad_senders.len(), "ban count");
    assert!(banned == traffic.bad_senders, "banned senders");

    if config
        .traffic
        .pass_window_us
        .is_some_and(|window| window < MIN_LATENCY_US)
    {
        assert_eq!(
            output.received.len(),
            traffic.raced_certs,
            "network certificates verified"
        );
    }

    for block in &traffic.blocks {
        for cert_type in [
            CertificateType::Notarize(*block),
            CertificateType::FinalizeFast(*block),
            CertificateType::Finalize(block.slot),
        ] {
            assert!(
                output.completed.contains(&cert_type),
                "{cert_type:?} was neither built nor received"
            );
        }
    }

    for cert in output.built.iter().chain(&output.received) {
        let unverified = UnverifiedCertificate {
            cert_type: cert.cert_type,
            signature: cert.signature,
            bitmap: cert.bitmap.clone(),
            shred_version: cluster.shred_version,
        };
        if let Err(err) = cluster.root_bank.verify_certificate(unverified, None) {
            panic!("{:?} does not verify: {err:?}", cert.cert_type);
        }
    }

    assert_eq!(traffic.footers.len(), traffic.blocks.len(), "footers");
    assert_eq!(output.valid_footers, traffic.footers.len(), "valid footers");

    // The sigverifier's threads send in any order, so sort for a stable listing.
    for (source, certs) in [("built", &output.built), ("network", &output.received)] {
        let mut certs: Vec<_> = certs.iter().collect();
        certs.sort_unstable_by_key(|cert| cert.cert_type);
        for cert in certs {
            println!(
                "CERT {source} {:?} {} {}",
                cert.cert_type,
                hex(&cert.bitmap),
                hex(&cert.signature.0)
            );
        }
    }
}

fn main() {
    let config = Config::from_args();
    let setup_start = Instant::now();
    let mut rng = Rng::new(config.seed);
    let cluster = Cluster::new(config.validators, &mut rng);
    let traffic = Traffic::new(&cluster, &config.traffic, &mut rng);
    let setup = setup_start.elapsed();
    let num_passes = traffic.passes.len();
    eprintln!(
        "agave-votor-bench: validators={} slots={} rounds={} threads={} pass_window_us={} \
         cert_copies={} cert_race={} bad_votes_per_slot={} seed={} datagrams={} passes={}",
        config.validators,
        config.traffic.slots,
        config.rounds,
        config.threads,
        config
            .traffic
            .pass_window_us
            .map_or("inf".to_string(), |window| window.to_string()),
        config.traffic.cert_copies,
        config.traffic.cert_race,
        config.traffic.bad_votes_per_slot,
        config.seed,
        traffic.datagrams.len(),
        num_passes,
    );

    let mut timings = Timings::default();
    let mut network_certs = 0;
    let mut built_certs = 0;
    for _ in 0..config.rounds {
        let output = run_round(&cluster, &traffic, &config, &mut timings);
        network_certs += output.received.len();
        built_certs += output.built.len();
        if config.check {
            check(&cluster, &traffic, &config, &output);
        }
    }

    let rounds = config.rounds.max(1);
    let total = timings.sigverify + timings.cert_build + timings.footer;
    println!("BENCH_PHASE:setup:{:.6}", setup.as_secs_f64());
    println!(
        "BENCH_PHASE:sigverify:{:.6}",
        timings.sigverify.as_secs_f64()
    );
    println!(
        "BENCH_PHASE:cert_build:{:.6}",
        timings.cert_build.as_secs_f64()
    );
    println!("BENCH_PHASE:footer:{:.6}", timings.footer.as_secs_f64());
    println!("BENCH_STAT:passes_per_round:{num_passes}");
    println!(
        "BENCH_STAT:datagrams_per_pass:{:.2}",
        traffic.datagrams.len() as f64 / num_passes as f64
    );
    println!(
        "BENCH_STAT:distinct_payloads_per_pass:{:.2}",
        traffic.distinct_payloads as f64 / num_passes as f64
    );
    println!(
        "BENCH_STAT:pass_us_mean:{:.1}",
        timings.sigverify.as_secs_f64() * 1e6 / (num_passes * rounds) as f64
    );
    println!(
        "BENCH_STAT:network_certs_verified_per_round:{:.1}",
        network_certs as f64 / rounds as f64
    );
    println!(
        "BENCH_STAT:certs_built_per_round:{:.1}",
        built_certs as f64 / rounds as f64
    );
    println!("BENCH_STAT:footers_per_round:{}", traffic.footers.len());
    if config.check {
        println!("CHECK OK");
    }
    println!("BENCH_RESULT:{:.6}", total.as_secs_f64());
}
