//! Builds the rooted history that the server reads: empty warm-up slots that
//! fill the recent blockhash window, then slots of transactions executed
//! through the unified scheduler, with transaction statuses written to the
//! blockstore.

use {
    crate::world::{TxKind, World, index},
    rand_chacha::ChaCha8Rng,
    solana_accounts_db::{
        accounts_db::{ACCOUNTS_DB_CONFIG_FOR_BENCHMARKS, AccountsDbConfig},
        accounts_index::AccountSecondaryIndexes,
    },
    solana_entry::entry::next_entry,
    solana_leader_schedule::SlotLeader,
    solana_ledger::blockstore::Blockstore,
    solana_pubkey::Pubkey,
    solana_rpc::{
        rpc::populate_blockstore_with_scheduler_pool_for_tests,
        transaction_status_service::{TransactionStatusService, TransactionStatusServiceConfig},
    },
    solana_runtime::{
        bank::Bank, bank_forks::BankForks, genesis_utils::GenesisConfigInfo,
        runtime_config::RuntimeConfig, transaction_execution::TransactionStatusSender,
    },
    solana_unified_scheduler_pool::DefaultSchedulerPool,
    solana_signature::Signature,
    solana_signer::Signer,
    std::{
        collections::HashMap,
        path::Path,
        sync::{
            Arc, RwLock,
            atomic::{AtomicBool, AtomicU64},
        },
    },
};

/// Enough empty slots to fill the window of 150 recent blockhashes.
pub const WARMUP_SLOTS: u64 = 150;
const MAX_RECENT_BLOCKHASHES: usize = 150;
const TRANSACTIONS_PER_ENTRY: usize = 64;

/// History transaction kinds with their weights in percent.
pub const HISTORY_MIX: [(TxKind, u64); 6] = [
    (TxKind::TokenTransfer, 20),
    (TxKind::SystemTransfer, 5),
    (TxKind::T22TransferMemo, 10),
    (TxKind::CreateAta(1), 5),
    (TxKind::Swap { fail: false }, 55),
    (TxKind::Swap { fail: true }, 5),
];

pub struct HistoryTransaction {
    pub signature: Signature,
    pub slot: u64,
    pub kind: TxKind,
}

pub struct History {
    pub bank_forks: Arc<RwLock<BankForks>>,
    pub tip: Arc<Bank>,
    pub transactions: Vec<HistoryTransaction>,
    pub transaction_slots: Vec<u64>,
    /// Number of history transactions that reference each active wallet.
    pub wallet_transactions: Vec<usize>,
    /// Number of history transactions that reference the Token program.
    pub token_program_transactions: usize,
    /// Transactions executed and failed, as counted by the banks.
    pub executed: u64,
    pub failed: u64,
}

pub fn new_bank(
    genesis: &GenesisConfigInfo,
    accounts_dir: &Path,
    account_indexes: &AccountSecondaryIndexes,
) -> Bank {
    let accounts_db_config = AccountsDbConfig {
        account_indexes: Some(account_indexes.clone()),
        ..ACCOUNTS_DB_CONFIG_FOR_BENCHMARKS
    };
    Bank::new_from_genesis(
        &genesis.genesis_config,
        Arc::new(RuntimeConfig::default()),
        vec![accounts_dir.to_path_buf()],
        None,
        accounts_db_config,
        None,
        None,
        Arc::default(),
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build(
    rng: &mut ChaCha8Rng,
    world: &World,
    genesis: &GenesisConfigInfo,
    root_bank: Bank,
    blockstore: &Arc<Blockstore>,
    transaction_slots: u64,
    transactions_per_slot: usize,
    handler_threads: usize,
) -> History {
    let leader = SlotLeader {
        id: genesis.validator_pubkey,
        vote_address: genesis.voting_keypair.pubkey(),
    };
    let mut blockhashes = vec![root_bank.last_blockhash()];
    let bank_forks = BankForks::new_rw_arc(root_bank);
    let mut parent = bank_forks.read().unwrap().root_bank();
    let active_index: HashMap<Pubkey, usize> = world
        .active
        .iter()
        .enumerate()
        .map(|(i, keypair)| (keypair.pubkey(), i))
        .collect();
    let total_weight: u64 = HISTORY_MIX.iter().map(|(_, weight)| weight).sum();
    let mut history = History {
        bank_forks: bank_forks.clone(),
        tip: parent.clone(),
        transactions: vec![],
        transaction_slots: vec![],
        wallet_transactions: vec![0; world.active.len()],
        token_program_transactions: 0,
        executed: 0,
        failed: 0,
    };
    // One transaction status service and one scheduler pool for all slots.
    // Extended metadata keeps log messages and inner instructions, as on an
    // RPC node with --enable-extended-tx-metadata-storage.
    let (transaction_status_sender, transaction_status_receiver) = crossbeam_channel::unbounded();
    let transaction_status_service = TransactionStatusService::new(
        transaction_status_receiver,
        Arc::new(AtomicU64::default()),
        Arc::new(AtomicU64::default()),
        blockstore.clone(),
        TransactionStatusServiceConfig {
            enable_rpc_transaction_history: true,
            enable_extended_tx_metadata_storage: true,
            ..TransactionStatusServiceConfig::default_for_tests()
        },
        Arc::new(AtomicBool::new(false)),
        None,
    );
    let pool = DefaultSchedulerPool::new_for_verification(
        Some(handler_threads),
        None,
        Some(TransactionStatusSender {
            sender: transaction_status_sender,
            dependency_tracker: None,
        }),
        None,
    );

    for slot in 1..=WARMUP_SLOTS + transaction_slots {
        let bank = Bank::new_from_parent(parent.clone(), leader.clone(), slot);
        let bank = bank_forks
            .write()
            .unwrap()
            .insert(bank)
            .clone_without_scheduler();
        let entries = if slot <= WARMUP_SLOTS {
            // One tick entry; the remaining ticks are registered below.
            vec![next_entry(&parent.last_blockhash(), 1, vec![])]
        } else {
            let window = &blockhashes[blockhashes.len().saturating_sub(MAX_RECENT_BLOCKHASHES)..];
            let transactions: Vec<_> = (0..transactions_per_slot)
                .map(|_| {
                    let mut pick = rng_below(rng, total_weight);
                    let kind = HISTORY_MIX
                        .iter()
                        .find(|(_, weight)| {
                            let found = pick < *weight;
                            pick = pick.saturating_sub(*weight);
                            found
                        })
                        .unwrap()
                        .0;
                    let owner = index(rng, world.active.len());
                    let blockhash = window[index(rng, window.len())];
                    let transaction = world.transaction(rng, kind, owner, blockhash);
                    for key in &transaction.message.account_keys {
                        if let Some(i) = active_index.get(key) {
                            history.wallet_transactions[*i] += 1;
                        }
                    }
                    if transaction
                        .message
                        .account_keys
                        .contains(&spl_token_interface::id())
                    {
                        history.token_program_transactions += 1;
                    }
                    history.transactions.push(HistoryTransaction {
                        signature: transaction.signatures[0],
                        slot,
                        kind,
                    });
                    transaction
                })
                .collect();
            history.transaction_slots.push(slot);
            let mut hash = parent.last_blockhash();
            transactions
                .chunks(TRANSACTIONS_PER_ENTRY)
                .map(|chunk| {
                    let entry = next_entry(&hash, 1, chunk.to_vec());
                    hash = entry.hash;
                    entry
                })
                .collect()
        };
        populate_blockstore_with_scheduler_pool_for_tests(entries, bank.clone(), blockstore, &pool);
        bank.fill_bank_with_ticks_for_tests();
        bank.freeze();
        history.executed += bank.executed_transaction_count();
        history.failed += bank.transaction_error_count();
        let block_time = genesis.genesis_config.creation_time + (slot / 5) as i64;
        blockstore.set_block_time(slot, block_time).unwrap();
        blockstore
            .set_block_height(slot, bank.block_height())
            .unwrap();
        blockhashes.push(bank.last_blockhash());
        parent = bank;
    }

    // Wait until every transaction status is written.
    transaction_status_service.join().unwrap();
    drop(pool);

    let tip_slot = parent.slot();
    bank_forks
        .write()
        .unwrap()
        .set_root(tip_slot, None, Some(tip_slot));
    parent.force_flush_accounts_cache();
    history.tip = parent;
    history
}

fn rng_below(rng: &mut ChaCha8Rng, n: u64) -> u64 {
    crate::world::range(rng, 0, n)
}
