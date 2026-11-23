//! Synthetic cluster and the consensus traffic one of its validators receives.

use {
    agave_votor::aggregate_accumulator::AggregateAccumulator,
    agave_votor_messages::{
        certificate::{CertSignature, Certificate, CertificateType},
        consensus_message::{Block, BlockId, VoteMessage},
        reward_certificate::{NUM_SLOTS_FOR_REWARD, NotarRewardCertificate, SkipRewardCertificate},
        stake::NonZeroStake,
        unverified_vote_message::UnverifiedCertificate,
        vote::Vote,
        wire::{VersionedWireConsensusMessage, get_vote_payload_to_sign},
    },
    agave_votor_transport::endpoint::Datagram,
    bytes::Bytes,
    rayon::prelude::*,
    solana_bls_signatures::signature::SignatureAffine,
    solana_clock::Slot,
    solana_entry::block_component::{BlockFinalizationCert, VotesAggregate},
    solana_epoch_schedule::EpochSchedule,
    solana_gossip::{cluster_info::ClusterInfo, contact_info::ContactInfo},
    solana_hash::Hash,
    solana_keypair::Keypair,
    solana_ledger::leader_schedule_cache::LeaderScheduleCache,
    solana_net_utils::SocketAddrSpace,
    solana_pubkey::Pubkey,
    solana_runtime::{
        bank::Bank,
        bank_forks::BankForks,
        epoch_stakes::BLSPubkeyToRankMap,
        genesis_utils::{ValidatorVoteKeypairs, create_genesis_config_with_vote_accounts},
    },
    solana_signer::Signer,
    std::{
        collections::{HashMap, HashSet},
        net::{Ipv4Addr, SocketAddr},
        ops::Range,
        sync::{Arc, RwLock},
    },
};

/// Total stake of the cluster in SOL, close to mainnet's.
const TOTAL_STAKE_SOL: f64 = 425_856_003.0;
const LAMPORTS_PER_SOL: u64 = 1_000_000_000;
/// Validator `i` gets a stake share proportional to `(i + STAKE_OFFSET)^-STAKE_EXPONENT`.
/// This matches mainnet's top-k stake shares within about 2 points.
const STAKE_OFFSET: f64 = 21.0;
const STAKE_EXPONENT: f64 = 1.56;

const SLOT_US: u64 = 200_000;
/// Delay from the start of a slot until the peers send their first vote.
const FIRST_VOTE_DELAY_US: u64 = 60_000;
/// Each peer has a fixed one-way latency, drawn from this range.
const MIN_LATENCY_US: u64 = 5_000;
const MAX_LATENCY_US: u64 = 100_000;
/// Every message gets an extra delay of up to this much.
const MAX_JITTER_US: u64 = 20_000;
/// A racing certificate copy arrives this long before the node's own threshold crossing.
const RACE_LEAD_US: u64 = 1_000;

/// Every this many slots, a few validators vote skip instead of notarize.
const SKIP_SLOT_INTERVAL: Slot = 8;
/// One in this many validators votes skip in those slots.
const SKIP_VOTER_INTERVAL: usize = 20;
/// Skip voters and senders of invalid votes are never among this many top-staked validators.
const TOP_RANKS: usize = 50;
/// Every this many slots, the footer carries a slow finalization (Finalize and Notarize).
const SLOW_FINALIZATION_INTERVAL: Slot = 4;

/// The same as the soft receive cap in the BLS sigverifier's `run()`.
const MAX_DATAGRAMS_PER_PASS: usize = 5_000;

/// A seeded xorshift64* generator, so that every architecture gets the same inputs.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Returns a value in `[low, high]`.
    fn range(&mut self, low: u64, high: u64) -> u64 {
        low + self.next_u64() % (high - low + 1)
    }

    /// Returns true with probability `p`.
    fn chance(&mut self, p: f64) -> bool {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < p
    }

    fn bytes(&mut self) -> [u8; 32] {
        let mut bytes = [0; 32];
        for chunk in bytes.as_chunks_mut::<8>().0 {
            *chunk = self.next_u64().to_le_bytes();
        }
        bytes
    }
}

/// Stakes in lamports, rounded to whole SOL so that they do not depend on the last bits of
/// `powf`.
fn power_law_stakes(num_validators: usize) -> Vec<u64> {
    let weights: Vec<f64> = (0..num_validators)
        .map(|i| (i as f64 + STAKE_OFFSET).powf(-STAKE_EXPONENT))
        .collect();
    let sum: f64 = weights.iter().sum();
    weights
        .iter()
        .map(|weight| (TOTAL_STAKE_SOL * weight / sum).round() as u64 * LAMPORTS_PER_SOL)
        .collect()
}

/// A cluster in epoch 0 with alpenglow enabled at genesis, seen by the validator at `my_rank`.
pub struct Cluster {
    /// Indexed by rank.
    pub validators: Vec<ValidatorVoteKeypairs>,
    pub bank_forks: Arc<RwLock<BankForks>>,
    pub root_bank: Arc<Bank>,
    pub rank_map: Arc<BLSPubkeyToRankMap>,
    pub my_rank: usize,
    pub cluster_info: Arc<ClusterInfo>,
    pub leader_schedule: Arc<LeaderScheduleCache>,
    pub shred_version: u16,
}

impl Cluster {
    pub fn new(num_validators: usize, rng: &mut Rng) -> Self {
        let seeds: Vec<_> = (0..num_validators)
            .map(|_| [rng.bytes(), rng.bytes(), rng.bytes()])
            .collect();
        let validators: Vec<_> = seeds
            .into_par_iter()
            .map(|[node, vote, stake]| {
                ValidatorVoteKeypairs::new(
                    Keypair::new_from_array(node),
                    Keypair::new_from_array(vote),
                    Keypair::new_from_array(stake),
                )
            })
            .collect();
        let mut genesis = create_genesis_config_with_vote_accounts(
            LAMPORTS_PER_SOL,
            &validators,
            power_law_stakes(num_validators),
        );
        genesis.genesis_config.epoch_schedule = EpochSchedule::without_warmup();
        let bank_forks = BankForks::new_rw_arc(Bank::new_for_tests(&genesis.genesis_config));
        let root_bank = bank_forks.read().unwrap().root_bank();
        let rank_map = root_bank.get_rank_map(1).expect("rank map").clone();
        assert_eq!(rank_map.len(), num_validators);

        let mut by_rank: Vec<Option<ValidatorVoteKeypairs>> =
            (0..num_validators).map(|_| None).collect();
        for validator in validators {
            let rank = rank_map
                .get_rank_for_vote_pubkey(&validator.vote_keypair.pubkey())
                .expect("every validator has a rank");
            by_rank[usize::from(*rank)] = Some(validator);
        }
        let validators: Vec<_> = by_rank.into_iter().map(Option::unwrap).collect();

        let my_rank = num_validators / 2;
        let my_keypair = validators[my_rank].node_keypair.insecure_clone();
        let contact_info = ContactInfo::new_localhost(&my_keypair.pubkey(), 0);
        let cluster_info = Arc::new(ClusterInfo::new(
            contact_info,
            Arc::new(my_keypair),
            SocketAddrSpace::Unspecified,
        ));
        let shred_version = cluster_info.my_shred_version();
        let leader_schedule = Arc::new(LeaderScheduleCache::new_from_bank(&root_bank));
        Self {
            validators,
            bank_forks,
            root_bank,
            rank_map,
            my_rank,
            cluster_info,
            leader_schedule,
            shred_version,
        }
    }

    fn stake(&self, rank: usize) -> NonZeroStake {
        self.rank_map
            .get_pubkey_stake_entry(rank)
            .expect("valid rank")
            .stake
    }

    fn node_pubkey(&self, rank: usize) -> Pubkey {
        self.validators[rank].node_keypair.pubkey()
    }

    fn vote_message(&self, vote: Vote, rank: usize, signature: SignatureAffine) -> VoteMessage {
        VoteMessage {
            vote,
            signature,
            rank: rank as u16,
            stake: self.stake(rank),
        }
    }
}

pub struct TrafficConfig {
    pub slots: Slot,
    /// Datagrams that arrive within this window of the first pending one share a pass.
    /// `None` means passes are only limited by `MAX_DATAGRAMS_PER_PASS`.
    pub pass_window_us: Option<u64>,
    /// Number of peers that send each certificate.
    pub cert_copies: usize,
    /// Fraction of certificates whose first copy arrives before the node builds them.
    pub cert_race: f64,
    pub bad_votes_per_slot: usize,
}

/// One call of the sigverifier.
pub struct Pass {
    /// Range in `Traffic::datagrams`.
    pub datagrams: Range<usize>,
    /// Footer certificates forwarded by blockstore, with their carrier slot.
    pub certificates: Vec<(Slot, UnverifiedCertificate)>,
}

/// A block footer as replay validates it, on a bank at the footer's slot.
pub struct Footer {
    pub bank: Bank,
    pub final_cert: BlockFinalizationCert,
    pub skip_reward_cert: Option<SkipRewardCertificate>,
    pub notar_reward_cert: Option<NotarRewardCertificate>,
}

pub struct Traffic {
    /// The block voted on in each slot.
    pub blocks: Vec<Block>,
    /// Datagrams from the peers in arrival order.
    pub datagrams: Vec<Datagram>,
    pub passes: Vec<Pass>,
    pub footers: Vec<Footer>,
    /// The node's own votes, which go to the pool directly.
    pub own_votes: HashMap<Vote, VoteMessage>,
    /// Every valid peer vote as (vote, rank).
    pub valid_votes: HashSet<(Vote, usize)>,
    /// Senders of invalid votes.
    pub bad_senders: HashSet<Pubkey>,
    /// Number of invalid votes. A sender can send more than one in different slots.
    pub num_bad_votes: usize,
    /// Types of the certificates the peers send.
    pub cert_types: HashSet<CertificateType>,
    /// Number of certificates whose first copy arrives before the node's own crossing.
    pub raced_certs: usize,
    /// Sum over passes of the number of distinct vote payloads in the pass.
    pub distinct_payloads: usize,
}

/// A datagram with the key it is sorted by.
struct Arrival {
    time_us: u64,
    slot: Slot,
    kind: u8,
    rank: usize,
    vote: Option<Vote>,
    datagram: Datagram,
}

fn serialize(message: &VersionedWireConsensusMessage) -> Bytes {
    Bytes::from(wincode::serialize(message).unwrap())
}

fn datagram(peer_pubkey: Pubkey, message: Bytes) -> Datagram {
    Datagram {
        peer_pubkey,
        // Never bound or connected to.
        peer_address: SocketAddr::from((Ipv4Addr::LOCALHOST, 1)),
        message,
    }
}

fn unverified(cert: &Certificate, shred_version: u16) -> UnverifiedCertificate {
    UnverifiedCertificate {
        cert_type: cert.cert_type,
        signature: cert.signature,
        bitmap: cert.bitmap.clone(),
        shred_version,
    }
}

fn votes_aggregate(cert: &Certificate) -> VotesAggregate {
    VotesAggregate::from_cert_signature(CertSignature {
        signature: cert.signature,
        bitmap: cert.bitmap.clone(),
    })
}

/// Returns the arrival time of the vote that brings the node's stake for `cert_type` to its
/// threshold, and a certificate made of the node's own vote and the votes up to that one.
fn threshold_crossing(
    cluster: &Cluster,
    cert_type: CertificateType,
    own_vote: &VoteMessage,
    arrivals: &[(u64, usize)],
    message: impl Fn(usize) -> VoteMessage,
) -> Option<(u64, Certificate)> {
    let total_stake = cluster.rank_map.total_stake();
    let mut accumulator = AggregateAccumulator::new(cluster.rank_map.len());
    accumulator.add_own_vote_message(own_vote).unwrap();
    for &(time_us, rank) in arrivals {
        accumulator.add_own_vote_message(&message(rank)).unwrap();
        if let Some(cert) = accumulator
            .try_build_base2_cert(cert_type, total_stake)
            .unwrap()
        {
            return Some((time_us, cert));
        }
    }
    None
}

/// Aggregates all `votes` into a reward certificate signature and bitmap.
fn reward_aggregate(
    cluster: &Cluster,
    votes: impl Iterator<Item = VoteMessage>,
) -> (solana_bls_signatures::SignatureCompressed, Vec<u8>) {
    let mut accumulator = AggregateAccumulator::new(cluster.rank_map.len());
    for vote in votes {
        accumulator.add_own_vote_message(&vote).unwrap();
    }
    accumulator.into_sig_and_ranks().unwrap()
}

/// Signatures of one validator for one slot.
struct SlotSignatures {
    /// On the notarize vote, or on the skip vote in a skip slot.
    first: SignatureAffine,
    finalize: Option<SignatureAffine>,
}

impl Traffic {
    pub fn new(cluster: &Cluster, config: &TrafficConfig, rng: &mut Rng) -> Self {
        let num_validators = cluster.validators.len();
        let my_rank = cluster.my_rank;
        let shred_version = cluster.shred_version;
        let peers: Vec<usize> = (0..num_validators).filter(|&r| r != my_rank).collect();
        let low_rank = TOP_RANKS.min(num_validators / 2);
        let skip_voters: HashSet<usize> = (low_rank..num_validators)
            .filter(|&r| r != my_rank && r % SKIP_VOTER_INTERVAL == SKIP_VOTER_INTERVAL - 1)
            .collect();
        let bad_vote_senders: Vec<usize> = (low_rank..num_validators)
            .filter(|r| *r != my_rank && !skip_voters.contains(r))
            .collect();
        assert!(
            config.bad_votes_per_slot == 0 || !bad_vote_senders.is_empty(),
            "too few validators for invalid votes"
        );
        let cert_senders: Vec<usize> = peers.iter().copied().take(config.cert_copies).collect();
        let latency: Vec<u64> = (0..num_validators)
            .map(|_| rng.range(MIN_LATENCY_US, MAX_LATENCY_US))
            .collect();
        let blocks: Vec<Block> = (1..=config.slots)
            .map(|slot| Block {
                slot,
                block_id: BlockId::from(Hash::new_from_array(rng.bytes())),
            })
            .collect();
        let is_skip_voter = |rank: usize, slot: Slot| {
            slot.is_multiple_of(SKIP_SLOT_INTERVAL) && skip_voters.contains(&rank)
        };
        let first_vote = |rank: usize, block: &Block| {
            if is_skip_voter(rank, block.slot) {
                Vote::new_skip_vote(block.slot)
            } else {
                Vote::new_notarization_vote(*block)
            }
        };

        // BLS signing dominates the setup, so do it in parallel.
        let signatures: Vec<Vec<SlotSignatures>> = cluster
            .validators
            .par_iter()
            .enumerate()
            .map(|(rank, validator)| {
                let sign = |vote| {
                    let payload = get_vote_payload_to_sign(vote, shred_version);
                    SignatureAffine::from(validator.bls_keypair.sign(&payload))
                };
                blocks
                    .iter()
                    .map(|block| SlotSignatures {
                        first: sign(first_vote(rank, block)),
                        finalize: (!is_skip_voter(rank, block.slot))
                            .then(|| sign(Vote::new_finalization_vote(block.slot))),
                    })
                    .collect()
            })
            .collect();

        let mut arrivals = Vec::new();
        let mut footer_certs = Vec::new();
        let mut footers = Vec::new();
        let mut own_votes = HashMap::new();
        let mut valid_votes = HashSet::new();
        let mut bad_senders = HashSet::new();
        let mut num_bad_votes = 0;
        let mut cert_types = HashSet::new();
        let mut raced_certs = 0;

        for (index, block) in blocks.iter().enumerate() {
            let slot = block.slot;
            let slot_start_us = SLOT_US * (slot - 1);
            let signature = |rank: usize| &signatures[rank][index];
            let notarize_vote = Vote::new_notarization_vote(*block);
            let finalize_vote = Vote::new_finalization_vote(slot);
            let skip_vote = Vote::new_skip_vote(slot);
            let own_notarize =
                cluster.vote_message(notarize_vote, my_rank, signature(my_rank).first);
            let own_finalize =
                cluster.vote_message(finalize_vote, my_rank, signature(my_rank).finalize.unwrap());
            own_votes.insert(notarize_vote, own_notarize.clone());
            own_votes.insert(finalize_vote, own_finalize.clone());

            let mut notarize_arrivals = Vec::new();
            let mut skip_ranks = Vec::new();
            for &rank in &peers {
                let vote = first_vote(rank, block);
                let time_us = slot_start_us
                    + FIRST_VOTE_DELAY_US
                    + latency[rank]
                    + rng.range(0, MAX_JITTER_US);
                let message = VersionedWireConsensusMessage::new_from_vote(
                    cluster.vote_message(vote, rank, signature(rank).first),
                    shred_version,
                );
                arrivals.push(Arrival {
                    time_us,
                    slot,
                    kind: 0,
                    rank,
                    vote: Some(vote),
                    datagram: datagram(cluster.node_pubkey(rank), serialize(&message)),
                });
                valid_votes.insert((vote, rank));
                if vote == skip_vote {
                    skip_ranks.push(rank);
                } else {
                    notarize_arrivals.push((time_us, rank));
                }
            }
            notarize_arrivals.sort_unstable();
            let notarize_message =
                |rank| cluster.vote_message(notarize_vote, rank, signature(rank).first);
            let notarize_crossing = threshold_crossing(
                cluster,
                CertificateType::Notarize(*block),
                &own_notarize,
                &notarize_arrivals,
                notarize_message,
            );
            let fast_crossing = threshold_crossing(
                cluster,
                CertificateType::FinalizeFast(*block),
                &own_notarize,
                &notarize_arrivals,
                notarize_message,
            );

            // Peers send finalize votes once they see the block notarized.
            let mut finalize_crossing = None;
            if let Some((notarized_us, _)) = &notarize_crossing {
                // The senders of invalid votes change every slot.
                let planted: HashSet<usize> = (0..config.bad_votes_per_slot)
                    .map(|k| {
                        let n = index * config.bad_votes_per_slot + k;
                        bad_vote_senders[n % bad_vote_senders.len()]
                    })
                    .collect();
                let mut finalize_arrivals = Vec::new();
                for &rank in &peers {
                    let Some(finalize_signature) = signature(rank).finalize else {
                        continue;
                    };
                    let time_us = notarized_us + 2 * latency[rank] + rng.range(0, MAX_JITTER_US);
                    // An invalid vote carries the sender's notarize signature: a valid point,
                    // but on the wrong message.
                    let is_bad = planted.contains(&rank);
                    let vote_signature = if is_bad {
                        signature(rank).first
                    } else {
                        finalize_signature
                    };
                    let message = VersionedWireConsensusMessage::new_from_vote(
                        cluster.vote_message(finalize_vote, rank, vote_signature),
                        shred_version,
                    );
                    arrivals.push(Arrival {
                        time_us,
                        slot,
                        kind: 1,
                        rank,
                        vote: Some(finalize_vote),
                        datagram: datagram(cluster.node_pubkey(rank), serialize(&message)),
                    });
                    if is_bad {
                        bad_senders.insert(cluster.node_pubkey(rank));
                        num_bad_votes += 1;
                    } else {
                        valid_votes.insert((finalize_vote, rank));
                        finalize_arrivals.push((time_us, rank));
                    }
                }
                finalize_arrivals.sort_unstable();
                finalize_crossing = threshold_crossing(
                    cluster,
                    CertificateType::Finalize(slot),
                    &own_finalize,
                    &finalize_arrivals,
                    |rank| {
                        cluster.vote_message(finalize_vote, rank, signature(rank).finalize.unwrap())
                    },
                );
            }

            // Every peer re-broadcasts each certificate once it has it.
            for (kind, crossing) in [
                (2, &notarize_crossing),
                (3, &fast_crossing),
                (4, &finalize_crossing),
            ] {
                let Some((crossing_us, cert)) = crossing else {
                    continue;
                };
                let mut copies: Vec<(u64, usize)> = cert_senders
                    .iter()
                    .map(|&rank| {
                        (
                            crossing_us + latency[rank] + rng.range(0, MAX_JITTER_US),
                            rank,
                        )
                    })
                    .collect();
                if !copies.is_empty() {
                    cert_types.insert(cert.cert_type);
                }
                if rng.chance(config.cert_race) && !copies.is_empty() {
                    raced_certs += 1;
                    let first = (0..copies.len()).min_by_key(|&i| copies[i]).unwrap();
                    copies[first].0 = crossing_us - RACE_LEAD_US;
                }
                let message = serialize(&VersionedWireConsensusMessage::new_from_cert(
                    cert.clone(),
                    shred_version,
                ));
                for (time_us, rank) in copies {
                    arrivals.push(Arrival {
                        time_us,
                        slot,
                        kind,
                        rank,
                        vote: None,
                        datagram: datagram(cluster.node_pubkey(rank), message.clone()),
                    });
                }
            }

            // The footer of block `slot + NUM_SLOTS_FOR_REWARD` finalizes this slot and carries
            // its reward certificates.
            let final_certs: Vec<&Certificate> =
                match (&notarize_crossing, &fast_crossing, &finalize_crossing) {
                    (Some((_, notarize)), _, Some((_, finalize)))
                        if slot.is_multiple_of(SLOW_FINALIZATION_INTERVAL) =>
                    {
                        vec![finalize, notarize]
                    }
                    (_, Some((_, fast)), _) => vec![fast],
                    _ => continue,
                };
            let final_cert = BlockFinalizationCert {
                slot,
                block_id: block.block_id,
                final_aggregate: votes_aggregate(final_certs[0]),
                notar_aggregate: final_certs.get(1).map(|cert| votes_aggregate(cert)),
            };
            let (signature_compressed, bitmap) = reward_aggregate(
                cluster,
                notarize_arrivals
                    .iter()
                    .map(|&(_, rank)| notarize_message(rank))
                    .chain(std::iter::once(own_notarize.clone())),
            );
            let notar_reward_cert = Some(
                NotarRewardCertificate::try_new(slot, block.block_id, signature_compressed, bitmap)
                    .unwrap(),
            );
            let skip_reward_cert = (!skip_ranks.is_empty()).then(|| {
                let (signature_compressed, bitmap) = reward_aggregate(
                    cluster,
                    skip_ranks
                        .iter()
                        .map(|&rank| cluster.vote_message(skip_vote, rank, signature(rank).first)),
                );
                SkipRewardCertificate::try_new(slot, signature_compressed, bitmap).unwrap()
            });
            let footer_slot = slot + NUM_SLOTS_FOR_REWARD;
            let leader = cluster
                .leader_schedule
                .slot_leader_at(footer_slot, Some(&cluster.root_bank))
                .expect("leader");
            let bank = Bank::new_from_parent(cluster.root_bank.clone(), leader, footer_slot);
            // Blockstore forwards the footer's final certificates to the sigverifier once the
            // block is complete.
            for cert in &final_certs {
                footer_certs.push((
                    SLOT_US * footer_slot,
                    footer_slot,
                    unverified(cert, shred_version),
                ));
            }
            footers.push(Footer {
                bank,
                final_cert,
                skip_reward_cert,
                notar_reward_cert,
            });
        }

        arrivals.sort_unstable_by_key(|a| (a.time_us, a.slot, a.kind, a.rank));
        footer_certs.sort_unstable_by_key(|(time_us, carrier_slot, cert)| {
            (*time_us, *carrier_slot, cert.cert_type)
        });
        let (passes, distinct_payloads) =
            make_passes(&arrivals, footer_certs, config.pass_window_us);
        let datagrams = arrivals.into_iter().map(|a| a.datagram).collect();
        Self {
            blocks,
            datagrams,
            passes,
            footers,
            own_votes,
            valid_votes,
            bad_senders,
            num_bad_votes,
            cert_types,
            raced_certs,
            distinct_payloads,
        }
    }
}

/// Splits the time-ordered inputs into sigverifier passes. A pass takes the datagrams that
/// arrive within the window of its first pending input, up to `MAX_DATAGRAMS_PER_PASS`, and the
/// footer certificates that arrive by then.
fn make_passes(
    arrivals: &[Arrival],
    footer_certs: Vec<(u64, Slot, UnverifiedCertificate)>,
    pass_window_us: Option<u64>,
) -> (Vec<Pass>, usize) {
    let max_datagrams = if pass_window_us == Some(0) {
        1
    } else {
        MAX_DATAGRAMS_PER_PASS
    };
    let mut footer_certs = footer_certs.into_iter().peekable();
    let mut passes = Vec::new();
    let mut distinct_payloads = 0;
    let mut next = 0;
    while next < arrivals.len() || footer_certs.peek().is_some() {
        let first_us = match (arrivals.get(next), footer_certs.peek()) {
            (Some(arrival), Some((cert_us, ..))) => arrival.time_us.min(*cert_us),
            (Some(arrival), None) => arrival.time_us,
            (None, Some((cert_us, ..))) => *cert_us,
            (None, None) => unreachable!(),
        };
        let window_end_us = pass_window_us.map_or(u64::MAX, |window| first_us + window);
        let start = next;
        while next < arrivals.len()
            && arrivals[next].time_us <= window_end_us
            && next - start < max_datagrams
        {
            next += 1;
        }
        let end_us = if next - start == max_datagrams {
            arrivals[next - 1].time_us
        } else {
            window_end_us
        };
        let mut certificates = Vec::new();
        while let Some((_, carrier_slot, cert)) =
            footer_certs.next_if(|(cert_us, ..)| *cert_us <= end_us)
        {
            certificates.push((carrier_slot, cert));
        }
        distinct_payloads += arrivals[start..next]
            .iter()
            .filter_map(|a| a.vote)
            .collect::<HashSet<_>>()
            .len();
        passes.push(Pass {
            datagrams: start..next,
            certificates,
        });
    }
    (passes, distinct_payloads)
}
