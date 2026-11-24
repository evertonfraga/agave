#[cfg(feature = "dev-context-only-utils")]
use qualifier_attr::qualifiers;
#[cfg(debug_assertions)]
use std::collections::HashSet;
use {
    crate::{
        bls_sigverifier::BAN_TIMEOUT, sig_verified_messages::VoteAggregate,
        stats::VoteVerificationStats, verified_batch::VerifiedBatch,
    },
    agave_votor_messages::{
        consensus_message::VoteMessage, stake::NonZeroStake,
        unverified_vote_message::UnverifiedVoteMessage, vote::Vote, wire::VotePayloadToSign,
    },
    agave_votor_transport::endpoint::BanSender,
    log::info,
    rayon::{
        ThreadPool, current_thread_index,
        iter::{
            IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
            ParallelIterator,
        },
    },
    solana_bls_signatures::{
        BlsError, PreparedHashedMessage, PubkeyProjective, SignatureProjective, VerifySignature,
        pubkey::{PopVerified, PubkeyAffine},
        signature::SignatureAffine,
    },
    solana_measure::{measure::Measure, measure_us},
    solana_pubkey::Pubkey,
    std::sync::Arc,
};

/// A batch of votes to verify.
///
/// Length of `batch` should be same length of `sender_vote_account_pubkeys`.
pub(crate) struct UnverifiedBatch {
    vote_payload_to_sign: VotePayloadToSign,
    batch: Vec<UnverifiedVotePayload>,
    sender_vote_account_pubkeys: Vec<Pubkey>,
}

impl UnverifiedBatch {
    pub(crate) fn new(
        vote_payload_to_sign: VotePayloadToSign,
        payload: UnverifiedVotePayload,
        sender_vote_account_pubkey: Pubkey,
    ) -> Self {
        Self {
            vote_payload_to_sign,
            batch: vec![payload],
            sender_vote_account_pubkeys: vec![sender_vote_account_pubkey],
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.batch.len()
    }

    pub(crate) fn push(
        &mut self,
        payload: UnverifiedVotePayload,
        sender_vote_account_pubkey: Pubkey,
    ) {
        self.batch.push(payload);
        self.sender_vote_account_pubkeys
            .push(sender_vote_account_pubkey);
    }

    /// Verifies the votes of the batch.
    ///
    /// `cached_prepared_msg` is the prepared message of the payload, if the sigverifier has it.
    /// Otherwise the payload is hashed and prepared here. Also returns the prepared message if at
    /// least one vote is valid, so that the sigverifier can cache it.
    pub(crate) fn verify(
        &mut self,
        max_validators: usize,
        cached_prepared_msg: Option<Arc<PreparedHashedMessage>>,
        ban_sender: &BanSender,
        thread_pool: &ThreadPool,
    ) -> (
        Option<VerifiedBatch>,
        VoteVerificationStats,
        Option<Arc<PreparedHashedMessage>>,
    ) {
        let mut stats = VoteVerificationStats::default();

        // no need to do optimistic verification when batch size == 1.
        if let [unverified_vote] = self.batch.as_slice() {
            let ((verification_result, sender_identity_pubkey, prepared_msg), time_us) =
                measure_us!({
                    let prepared_msg = cached_prepared_msg
                        .unwrap_or_else(|| prepare_msg(&self.vote_payload_to_sign));
                    let sender_identity_pubkey = unverified_vote.sender_identity_pubkey;
                    (
                        unverified_vote.verify(max_validators, &prepared_msg),
                        sender_identity_pubkey,
                        prepared_msg,
                    )
                });
            stats.fn_verify_individual_votes_stats.add_sample(time_us);
            return match verification_result {
                Ok(vote_aggregate) => {
                    stats.num_individual_verified += 1;
                    let pubkeys = std::mem::take(&mut self.sender_vote_account_pubkeys);
                    (
                        Some(VerifiedBatch::new(
                            Vote::from(self.vote_payload_to_sign),
                            vec![vote_aggregate],
                            pubkeys,
                        )),
                        stats,
                        Some(prepared_msg),
                    )
                }
                Err(error) => {
                    ban_invalid_vote_sender(ban_sender, &mut stats, sender_identity_pubkey, error);
                    (None, stats, None)
                }
            };
        }

        // Try optimistic verification - fast to verify, but cannot identify invalid votes
        let (res, prepared_msg) = verify_votes_optimistic(
            &self.vote_payload_to_sign,
            cached_prepared_msg,
            &self.batch,
            &mut stats,
            thread_pool,
        );

        let sender_vote_account_pubkeys = std::mem::take(&mut self.sender_vote_account_pubkeys);
        match res {
            Some(signature) => {
                stats.optimistic_verification_succeeded += 1;
                stats.optimistic_batch.add_sample(self.batch.len() as u64);
                let vote_aggregate = VoteAggregate::new_from_verified_votes(
                    max_validators,
                    self.vote_payload_to_sign,
                    self.batch.iter().map(|v| (v.rank, v.stake)),
                    signature,
                );
                (
                    Some(VerifiedBatch::new(
                        Vote::from(self.vote_payload_to_sign),
                        vec![vote_aggregate],
                        sender_vote_account_pubkeys,
                    )),
                    stats,
                    Some(prepared_msg),
                )
            }
            None => {
                // Fallback to individual verification
                stats.optimistic_verification_failed += 1;
                let ((verified_batch, invalid_remote_pubkeys), time_us) =
                    measure_us!(verify_individual_votes(
                        Vote::from(self.vote_payload_to_sign),
                        max_validators,
                        &self.batch,
                        sender_vote_account_pubkeys,
                        &prepared_msg,
                        thread_pool
                    ));
                if let Some(b) = &verified_batch {
                    stats.num_individual_verified += b.len();
                }
                for (sender_identity_pubkey, error) in invalid_remote_pubkeys {
                    ban_invalid_vote_sender(ban_sender, &mut stats, sender_identity_pubkey, error);
                }
                stats.fn_verify_individual_votes_stats.add_sample(time_us);
                let prepared_msg = verified_batch.is_some().then_some(prepared_msg);
                (verified_batch, stats, prepared_msg)
            }
        }
    }
}

/// Hashes the payload to the curve and prepares the result for pairing.
fn prepare_msg(vote_payload_to_sign: &VotePayloadToSign) -> Arc<PreparedHashedMessage> {
    let hashed_msg = vote_payload_to_sign.to_hashed_msg();
    Arc::new(PreparedHashedMessage::from_hashed_message(&hashed_msg))
}

fn ban_invalid_vote_sender(
    ban_sender: &BanSender,
    stats: &mut VoteVerificationStats,
    sender_identity_pubkey: Pubkey,
    error: BlsError,
) {
    stats.banning_validator += 1;
    ban_sender.ban(sender_identity_pubkey, BAN_TIMEOUT);
    info!(
        "bls_vote_sigverify: banned sender={sender_identity_pubkey} due to failed verification \
         {error:?}"
    );
}

#[cfg_attr(feature = "dev-context-only-utils", qualifiers(pub))]
/// Attempts aggregate BLS verification across the full vote set.
///
/// This fast path aggregates all vote signatures and the public keys for each
/// distinct vote payload, minimizing the number of pairing operations needed
/// for verification. When aggregation or aggregate verification fails, the
/// caller falls back to individual vote verification so invalid votes can be
/// identified precisely.
///
/// Returns the aggregate signature if the optimistic verification succeeds,
/// together with the prepared message of the payload, which the fallback path
/// reuses. The prepared message is `cached_prepared_msg` if the caller has one.
/// Otherwise it is computed while the signatures and public keys are
/// aggregated.
fn verify_votes_optimistic(
    vote_payload_to_sign: &VotePayloadToSign,
    cached_prepared_msg: Option<Arc<PreparedHashedMessage>>,
    unverified_votes: &[UnverifiedVotePayload],
    stats: &mut VoteVerificationStats,
    thread_pool: &ThreadPool,
) -> (Option<SignatureProjective>, Arc<PreparedHashedMessage>) {
    #[cfg(debug_assertions)]
    {
        let deduped = unverified_votes
            .iter()
            .map(|v| &v.vote_message)
            .collect::<HashSet<_>>();
        assert_eq!(deduped.len(), unverified_votes.len());
    }

    let mut measure = Measure::start("verify_votes_optimistic");

    // For BLS verification, minimizing the expensive pairing operation is key.
    // Each BLS signature verification requires two pairings.
    //
    // However, the BLS verification formula allows us to:
    // 1. Aggregate all signatures into a single signature.
    // 2. Aggregate public keys for each unique message.
    //
    // By verifying the aggregated signature against the aggregated public keys,
    // the number of pairings required is reduced to (1 + number of distinct messages).
    let (signature_result, (pubkey_result, prepared_msg)) = thread_pool.join(
        || aggregate_signatures(unverified_votes),
        || match cached_prepared_msg {
            Some(prepared_msg) => (aggregate_pubkeys_by_payload(unverified_votes), prepared_msg),
            None => thread_pool.join(
                || aggregate_pubkeys_by_payload(unverified_votes),
                || prepare_msg(vote_payload_to_sign),
            ),
        },
    );

    let Ok(aggregate_signature) = signature_result else {
        return (None, prepared_msg);
    };

    let Ok(aggregate_pubkey) = pubkey_result else {
        return (None, prepared_msg);
    };

    let verified = aggregate_pubkey.verify_signature_prepared(&aggregate_signature, &prepared_msg);

    measure.stop();
    stats
        .fn_verify_votes_optimistic_stats
        .add_sample(measure.as_us());
    (
        verified.is_ok().then_some(aggregate_signature),
        prepared_msg,
    )
}

#[cfg_attr(feature = "dev-context-only-utils", qualifiers(pub))]
fn aggregate_signatures(votes: &[UnverifiedVotePayload]) -> Result<SignatureProjective, BlsError> {
    debug_assert!(current_thread_index().is_some());
    let signatures = votes.par_iter().map(|v| &v.vote_message.signature);
    // TODO(sam): Currently, `par_aggregate` performs full validation
    // (on-curve + subgroup check) for every signature. Since the subgroup
    // check is expensive, we can use an `unchecked` deserialization here
    // (performing only the cheap on-curve check) and rely on a single subgroup
    // check on the final aggregated signature. This should save more than 80%
    // of the time for signature aggregation.
    SignatureProjective::par_aggregate(signatures)
}

#[cfg_attr(feature = "dev-context-only-utils", qualifiers(pub))]
fn aggregate_pubkeys_by_payload(
    votes: &[UnverifiedVotePayload],
) -> Result<PopVerified<PubkeyProjective>, BlsError> {
    debug_assert!(current_thread_index().is_some());
    // converting aggregate pubkey to `PopVerified` is safe here
    // since the pubkeys are all PoP verified in the vote account
    PubkeyProjective::par_aggregate(votes.into_par_iter().map(|v| &v.sender_bls_pubkey))
        .map(|agg| unsafe { PopVerified::new_unchecked(*agg) })
}

/// Verifies votes individually on a thread pool.
///
/// Returns:
/// - `Vec<VotePayload>`: votes that passed verification.
/// - `Vec<Pubkey>`: senders' identity pubkeys for votes that failed verification.
#[cfg_attr(feature = "dev-context-only-utils", qualifiers(pub))]
fn verify_individual_votes(
    vote: Vote,
    max_validators: usize,
    unverified_votes: &[UnverifiedVotePayload],
    sender_vote_account_pubkeys: Vec<Pubkey>,
    prepared_msg: &PreparedHashedMessage,
    thread_pool: &ThreadPool,
) -> (Option<VerifiedBatch>, Vec<(Pubkey, BlsError)>) {
    let (aggregates, sender_vote_account_pubkeys, failed) = thread_pool.install(|| {
        unverified_votes
            .into_par_iter()
            .zip(sender_vote_account_pubkeys)
            .fold(
                || (vec![], vec![], vec![]),
                |(mut verified, mut sender_vote_account_pubkeys, mut failed),
                 (unverified_vote, sender_vote_account_pubkey)| {
                    let sender_identity_pubkey = unverified_vote.sender_identity_pubkey;
                    match unverified_vote.verify(max_validators, prepared_msg) {
                        Ok(aggregate) => {
                            verified.push(aggregate);
                            sender_vote_account_pubkeys.push(sender_vote_account_pubkey);
                        }
                        Err(e) => failed.push((sender_identity_pubkey, e)),
                    }
                    (verified, sender_vote_account_pubkeys, failed)
                },
            )
            .reduce(
                || (vec![], vec![], vec![]),
                |mut left, mut right| {
                    left.0.append(&mut right.0);
                    left.1.append(&mut right.1);
                    left.2.append(&mut right.2);
                    left
                },
            )
    });
    (
        (!aggregates.is_empty())
            .then(|| VerifiedBatch::new(vote, aggregates, sender_vote_account_pubkeys)),
        failed,
    )
}

/// [`VoteMessage`] along with other information needed to sig verify it.
#[cfg_attr(feature = "dev-context-only-utils", qualifiers(pub))]
#[derive(Clone, Debug)]
pub(super) struct UnverifiedVotePayload {
    pub vote_message: UnverifiedVoteMessage,
    pub sender_bls_pubkey: PopVerified<PubkeyAffine>,
    pub sender_identity_pubkey: Pubkey,
    pub rank: u16,
    pub stake: NonZeroStake,
}

impl UnverifiedVotePayload {
    fn verify(
        &self,
        max_validators: usize,
        prepared_msg: &PreparedHashedMessage,
    ) -> Result<VoteAggregate, BlsError> {
        let signature = SignatureAffine::try_from(self.vote_message.signature)?;
        self.sender_bls_pubkey
            .verify_signature_prepared(&signature, prepared_msg)?;
        let vote_msg = VoteMessage {
            vote: self.vote_message.vote,
            signature,
            rank: self.rank,
            stake: self.stake,
        };
        Ok(VoteAggregate::new_from_verified_vote(
            max_validators,
            vote_msg,
        ))
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use {
        super::*,
        agave_votor_messages::wire::get_vote_payload_to_sign,
        agave_votor_transport::endpoint::{BanCommand, stub_ban_channel_for_tests},
        rayon::ThreadPoolBuilder,
        solana_bls_signatures::{BLS_SIGNATURE_AFFINE_SIZE, Keypair as BLSKeypair, Signature},
    };

    const NUM_VALIDATORS: usize = 6;
    const SHRED_VERSION: u16 = 42;

    #[derive(Clone, Copy)]
    enum Signer {
        /// Signs the payload of the vote.
        Valid,
        /// Signs the payload of another vote.
        OtherPayload,
        /// Sends bytes that are not a curve point.
        NotAPoint,
    }

    #[derive(Debug, PartialEq)]
    struct Outcome {
        verified_batch: Option<VerifiedBatch>,
        optimistic_verification_succeeded: u64,
        optimistic_verification_failed: u64,
        num_individual_verified: usize,
        banned: Vec<Pubkey>,
    }

    struct TestValidators {
        bls_keypairs: Vec<BLSKeypair>,
        identity_pubkeys: Vec<Pubkey>,
        vote_account_pubkeys: Vec<Pubkey>,
        thread_pool: ThreadPool,
    }

    impl TestValidators {
        fn new() -> Self {
            Self {
                bls_keypairs: (0..NUM_VALIDATORS).map(|_| BLSKeypair::new()).collect(),
                identity_pubkeys: (0..NUM_VALIDATORS).map(|_| Pubkey::new_unique()).collect(),
                vote_account_pubkeys: (0..NUM_VALIDATORS).map(|_| Pubkey::new_unique()).collect(),
                thread_pool: ThreadPoolBuilder::new().num_threads(4).build().unwrap(),
            }
        }

        /// Verifies a batch of votes for `vote` from the ranks in `signers`.
        fn verify(
            &self,
            vote: Vote,
            signers: &[(usize, Signer)],
            cached_prepared_msg: Option<Arc<PreparedHashedMessage>>,
        ) -> (Outcome, Option<Arc<PreparedHashedMessage>>) {
            let payload = get_vote_payload_to_sign(vote, SHRED_VERSION);
            let other_payload =
                get_vote_payload_to_sign(Vote::new_skip_vote(vote.slot() + 1), SHRED_VERSION);
            let mut votes = signers.iter().map(|&(rank, signer)| {
                let bls_keypair = &self.bls_keypairs[rank];
                let signature = match signer {
                    Signer::Valid => bls_keypair.sign(&payload).into(),
                    Signer::OtherPayload => bls_keypair.sign(&other_payload).into(),
                    Signer::NotAPoint => Signature([0; BLS_SIGNATURE_AFFINE_SIZE]),
                };
                let unverified_vote = UnverifiedVotePayload {
                    vote_message: UnverifiedVoteMessage {
                        vote,
                        signature,
                        shred_version: SHRED_VERSION,
                    },
                    sender_bls_pubkey: bls_keypair.public,
                    sender_identity_pubkey: self.identity_pubkeys[rank],
                    rank: rank as u16,
                    stake: NonZeroStake::new(100 + rank as u64).unwrap(),
                };
                (unverified_vote, self.vote_account_pubkeys[rank])
            });
            let (first_vote, first_vote_account_pubkey) = votes.next().unwrap();
            let mut batch = UnverifiedBatch::new(
                VotePayloadToSign::new_from_vote(vote, SHRED_VERSION),
                first_vote,
                first_vote_account_pubkey,
            );
            for (unverified_vote, vote_account_pubkey) in votes {
                batch.push(unverified_vote, vote_account_pubkey);
            }

            let (ban_sender, mut ban_receiver) = stub_ban_channel_for_tests(NUM_VALIDATORS);
            let (verified_batch, stats, prepared_msg) = batch.verify(
                NUM_VALIDATORS,
                cached_prepared_msg,
                &ban_sender,
                &self.thread_pool,
            );
            let mut banned = Vec::new();
            while let Ok(BanCommand { peer, .. }) = ban_receiver.try_recv() {
                banned.push(peer);
            }
            assert_eq!(stats.banning_validator.0, banned.len() as u64);
            let outcome = Outcome {
                verified_batch,
                optimistic_verification_succeeded: stats.optimistic_verification_succeeded.0,
                optimistic_verification_failed: stats.optimistic_verification_failed.0,
                num_individual_verified: stats.num_individual_verified.0,
                banned,
            };
            (outcome, prepared_msg)
        }
    }

    /// Verifies the votes without a cached prepared message, with an independently prepared
    /// one, and with the one that the first verification returned, and checks that all three give
    /// the same result.
    fn check_cached_prepared_msg(
        validators: &TestValidators,
        signers: &[(usize, Signer)],
        expected_verified: &[usize],
        expected_banned: &[usize],
    ) {
        let vote = Vote::new_skip_vote(42);
        let (outcome, returned_msg) = validators.verify(vote, signers, None);
        let num_aggregates = outcome
            .verified_batch
            .as_ref()
            .map_or(0, VerifiedBatch::len);
        let num_verified = if outcome.optimistic_verification_succeeded > 0 {
            // One aggregate holds all the votes.
            assert_eq!(num_aggregates, 1);
            assert_eq!(outcome.num_individual_verified, 0);
            signers.len()
        } else {
            // One aggregate for each valid vote.
            assert_eq!(num_aggregates, outcome.num_individual_verified);
            outcome.num_individual_verified
        };
        assert_eq!(num_verified, expected_verified.len());
        let expected_banned: Vec<_> = expected_banned
            .iter()
            .map(|&rank| validators.identity_pubkeys[rank])
            .collect();
        assert_eq!(outcome.banned, expected_banned);
        // The message is returned for caching only if at least one vote is valid.
        assert_eq!(returned_msg.is_some(), !expected_verified.is_empty());

        let cached_msg = Arc::new(PreparedHashedMessage::new(&get_vote_payload_to_sign(
            vote,
            SHRED_VERSION,
        )));
        let (cached_outcome, cached_returned_msg) =
            validators.verify(vote, signers, Some(cached_msg.clone()));
        assert_eq!(cached_outcome, outcome);
        assert_eq!(cached_returned_msg.is_some(), returned_msg.is_some());
        if let Some(cached_returned_msg) = cached_returned_msg {
            assert!(Arc::ptr_eq(&cached_returned_msg, &cached_msg));
        }

        if let Some(returned_msg) = returned_msg {
            let (reused_outcome, _) = validators.verify(vote, signers, Some(returned_msg));
            assert_eq!(reused_outcome, outcome);
        }
    }

    #[test]
    fn test_single_vote_with_cached_prepared_msg() {
        let validators = TestValidators::new();
        check_cached_prepared_msg(&validators, &[(0, Signer::Valid)], &[0], &[]);
        check_cached_prepared_msg(&validators, &[(1, Signer::OtherPayload)], &[], &[1]);
        check_cached_prepared_msg(&validators, &[(2, Signer::NotAPoint)], &[], &[2]);
    }

    #[test]
    fn test_optimistic_verification_with_cached_prepared_msg() {
        let validators = TestValidators::new();
        let signers: Vec<_> = (0..NUM_VALIDATORS)
            .map(|rank| (rank, Signer::Valid))
            .collect();
        check_cached_prepared_msg(&validators, &signers, &[0, 1, 2, 3, 4, 5], &[]);
    }

    #[test]
    fn test_fallback_verification_with_cached_prepared_msg() {
        let validators = TestValidators::new();
        check_cached_prepared_msg(
            &validators,
            &[
                (0, Signer::Valid),
                (1, Signer::OtherPayload),
                (2, Signer::Valid),
                (3, Signer::OtherPayload),
            ],
            &[0, 2],
            &[1, 3],
        );
        // The signatures cannot be aggregated.
        check_cached_prepared_msg(
            &validators,
            &[
                (3, Signer::Valid),
                (4, Signer::NotAPoint),
                (5, Signer::Valid),
            ],
            &[3, 5],
            &[4],
        );
        // No vote is valid, so nothing is returned for caching.
        check_cached_prepared_msg(
            &validators,
            &[
                (0, Signer::OtherPayload),
                (2, Signer::NotAPoint),
                (4, Signer::OtherPayload),
            ],
            &[],
            &[0, 2, 4],
        );
    }
}
