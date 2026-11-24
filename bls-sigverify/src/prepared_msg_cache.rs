use {
    agave_votor_messages::{reward_certificate::NUM_SLOTS_FOR_REWARD, wire::VotePayloadToSign},
    solana_bls_signatures::PreparedHashedMessage,
    solana_clock::Slot,
    std::{collections::HashMap, sync::Arc},
};

/// The maximum number of vote payloads that [`PreparedMsgCache`] holds.
///
/// Each prepared message owns a pairing preparation table of 19,584 bytes. With the hashed point
/// and the allocation, an entry takes about 19.8 KB, so a full cache holds about 5 MB.
const MAX_PREPARED_MSGS: usize = 256;

/// Prepared messages of recent vote payloads.
///
/// Votes for the same payload are verified in many sigverifier passes. With the prepared message
/// at hand, a pass does not have to hash the payload to the curve and prepare it for pairing
/// again. The sigverifier only inserts payloads that had at least one valid vote.
#[derive(Default)]
pub(crate) struct PreparedMsgCache {
    map: HashMap<VotePayloadToSign, Arc<PreparedHashedMessage>>,
}

impl PreparedMsgCache {
    /// Returns the prepared message of `vote_payload_to_sign`, if the cache has it.
    pub(crate) fn get(
        &self,
        vote_payload_to_sign: &VotePayloadToSign,
    ) -> Option<Arc<PreparedHashedMessage>> {
        self.map.get(vote_payload_to_sign).cloned()
    }

    /// Inserts the prepared message of `vote_payload_to_sign`, unless the cache already has one.
    ///
    /// When the cache is full, evicts the payload with the lowest slot if that slot is lower than
    /// the slot of `vote_payload_to_sign`, and otherwise does not insert.
    pub(crate) fn insert(
        &mut self,
        vote_payload_to_sign: VotePayloadToSign,
        prepared_msg: Arc<PreparedHashedMessage>,
    ) {
        if self.map.contains_key(&vote_payload_to_sign) {
            return;
        }
        if self.map.len() >= MAX_PREPARED_MSGS {
            match self.map.keys().copied().min_by_key(VotePayloadToSign::slot) {
                Some(lowest) if lowest.slot() < vote_payload_to_sign.slot() => {
                    self.map.remove(&lowest);
                }
                _ => return,
            }
        }
        self.map.insert(vote_payload_to_sign, prepared_msg);
    }

    /// Removes the payloads of slots that the vote pool no longer keeps votes for.
    pub(crate) fn prune(&mut self, root_slot: Slot) {
        let slot_to_keep = root_slot.saturating_sub(NUM_SLOTS_FOR_REWARD);
        self.map
            .retain(|vote_payload_to_sign, _| vote_payload_to_sign.slot() >= slot_to_keep);
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use {
        super::*,
        agave_votor_messages::vote::Vote,
        std::sync::{LazyLock, Weak},
    };

    const SHRED_VERSION: u16 = 42;

    // Preparing a message is slow in debug builds, so the tests share one.
    static PREPARED_MSG: LazyLock<PreparedHashedMessage> =
        LazyLock::new(|| PreparedHashedMessage::new(b"prepared message"));

    /// Returns a new allocation, so that a test can tell the entries apart.
    fn new_prepared_msg() -> Arc<PreparedHashedMessage> {
        Arc::new(PREPARED_MSG.clone())
    }

    fn skip_payload(slot: Slot) -> VotePayloadToSign {
        VotePayloadToSign::new_from_vote(Vote::new_skip_vote(slot), SHRED_VERSION)
    }

    /// Returns a payload for a new block in `slot`.
    fn notar_payload(slot: Slot) -> VotePayloadToSign {
        VotePayloadToSign::new_from_vote(Vote::new_unique_notar(slot), SHRED_VERSION)
    }

    /// Fills the cache with one payload for each of the slots `first_slot..`.
    fn new_full_cache(first_slot: Slot) -> PreparedMsgCache {
        let mut cache = PreparedMsgCache::default();
        for slot in first_slot..first_slot + MAX_PREPARED_MSGS as Slot {
            cache.insert(skip_payload(slot), new_prepared_msg());
        }
        assert_eq!(cache.map.len(), MAX_PREPARED_MSGS);
        cache
    }

    #[test]
    fn test_get_miss_and_hit() {
        let mut cache = PreparedMsgCache::default();
        let payload = skip_payload(10);
        assert!(cache.get(&payload).is_none());

        let prepared_msg = new_prepared_msg();
        cache.insert(payload, prepared_msg.clone());
        assert!(Arc::ptr_eq(&cache.get(&payload).unwrap(), &prepared_msg));

        // The same vote with another shred version is another payload.
        let other_shred_version =
            VotePayloadToSign::new_from_vote(Vote::new_skip_vote(10), SHRED_VERSION + 1);
        assert!(cache.get(&other_shred_version).is_none());
        assert!(cache.get(&skip_payload(11)).is_none());
    }

    #[test]
    fn test_insert_keeps_cached_msg() {
        let mut cache = PreparedMsgCache::default();
        let payload = skip_payload(10);
        let first = new_prepared_msg();
        cache.insert(payload, first.clone());
        cache.insert(payload, new_prepared_msg());
        assert!(Arc::ptr_eq(&cache.get(&payload).unwrap(), &first));
        assert_eq!(cache.map.len(), 1);
    }

    #[test]
    fn test_prune_matches_vote_pool() {
        let mut cache = PreparedMsgCache::default();
        for slot in 0..30 {
            cache.insert(skip_payload(slot), new_prepared_msg());
        }
        // Nothing is below the first slot that the vote pool keeps.
        cache.prune(NUM_SLOTS_FOR_REWARD);
        assert_eq!(cache.map.len(), 30);

        let root_slot = 20;
        cache.prune(root_slot);
        for slot in 0..30 {
            assert_eq!(
                cache.get(&skip_payload(slot)).is_some(),
                slot >= root_slot - NUM_SLOTS_FOR_REWARD,
                "slot {slot}"
            );
        }
    }

    #[test]
    fn test_full_cache_evicts_lowest_slot() {
        let mut cache = new_full_cache(100);
        let first_payload = skip_payload(100);

        // A payload in a higher slot replaces the payload in the lowest slot.
        let new_payload = notar_payload(100 + MAX_PREPARED_MSGS as Slot);
        cache.insert(new_payload, new_prepared_msg());
        assert_eq!(cache.map.len(), MAX_PREPARED_MSGS);
        assert!(cache.get(&new_payload).is_some());
        assert!(cache.get(&first_payload).is_none());
        assert!(cache.get(&skip_payload(101)).is_some());

        // A payload in a slot that is not higher than the lowest one is not inserted.
        for slot in [0, 100, 101] {
            let payload = notar_payload(slot);
            cache.insert(payload, new_prepared_msg());
            assert!(cache.get(&payload).is_none(), "slot {slot}");
            assert_eq!(cache.map.len(), MAX_PREPARED_MSGS);
        }
        assert!(cache.get(&skip_payload(101)).is_some());

        // A payload that the cache already has does not evict another one.
        cache.insert(skip_payload(200), new_prepared_msg());
        assert_eq!(cache.map.len(), MAX_PREPARED_MSGS);
        assert!(cache.get(&skip_payload(101)).is_some());
    }

    #[test]
    fn test_full_cache_in_one_slot() {
        let mut cache = PreparedMsgCache::default();
        let payloads: Vec<_> = (0..MAX_PREPARED_MSGS).map(|_| notar_payload(7)).collect();
        for payload in &payloads {
            cache.insert(*payload, new_prepared_msg());
        }
        let payload = notar_payload(7);
        cache.insert(payload, new_prepared_msg());
        assert!(cache.get(&payload).is_none());

        let payload = notar_payload(8);
        cache.insert(payload, new_prepared_msg());
        assert!(cache.get(&payload).is_some());
        assert_eq!(cache.map.len(), MAX_PREPARED_MSGS);
        assert_eq!(
            payloads.iter().filter(|p| cache.get(p).is_some()).count(),
            MAX_PREPARED_MSGS - 1
        );
    }

    #[test]
    fn test_memory_cap() {
        // A prepared message owns a table of 68 Fp6 elements of 288 bytes each.
        const TABLE_BYTES: usize = 68 * 288;
        let entry_bytes = TABLE_BYTES + size_of::<PreparedHashedMessage>();
        assert!(MAX_PREPARED_MSGS * entry_bytes < 5_100_000);

        let mut cache = PreparedMsgCache::default();
        let mut inserted = Vec::<Weak<PreparedHashedMessage>>::new();
        // Payloads in rising slots, then in falling slots, then many in the same slot.
        let slots = (1_000..1_000 + 2 * MAX_PREPARED_MSGS as Slot)
            .chain((0..1_000).rev().step_by(4))
            .chain(std::iter::repeat_n(5_000, MAX_PREPARED_MSGS));
        for slot in slots {
            let prepared_msg = new_prepared_msg();
            inserted.push(Arc::downgrade(&prepared_msg));
            cache.insert(notar_payload(slot), prepared_msg);
            assert!(cache.map.len() <= MAX_PREPARED_MSGS);
        }
        assert_eq!(cache.map.len(), MAX_PREPARED_MSGS);
        // Messages that were evicted or not inserted are freed.
        let live = inserted
            .iter()
            .filter(|prepared_msg| prepared_msg.strong_count() > 0)
            .count();
        assert_eq!(live, MAX_PREPARED_MSGS);

        cache.prune(u64::MAX);
        assert!(cache.map.is_empty());
        assert!(inserted.iter().all(|msg| msg.strong_count() == 0));
    }
}
