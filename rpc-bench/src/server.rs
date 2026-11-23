//! Starts `JsonRpcService` over the frozen history, the way the validator
//! does, bound to 127.0.0.1 only.

use {
    crate::history::History,
    solana_gossip::{cluster_info::ClusterInfo, contact_info::ContactInfo},
    solana_hash::Hash,
    solana_keypair::Keypair,
    solana_ledger::{blockstore::Blockstore, leader_schedule_cache::LeaderScheduleCache},
    solana_net_utils::SocketAddrSpace,
    solana_rpc::{
        max_slots::MaxSlots,
        optimistically_confirmed_bank_tracker::OptimisticallyConfirmedBank,
        rpc::{JsonRpcConfig, create_validator_exit},
        rpc_service::{JsonRpcService, JsonRpcServiceConfig, RpcTpuClientArgs},
    },
    solana_runtime::commitment::{BlockCommitmentCache, CommitmentSlots},
    solana_send_transaction_service::send_transaction_service,
    solana_signer::Signer,
    std::{
        collections::HashMap,
        net::{Ipv4Addr, SocketAddr, TcpListener, UdpSocket},
        path::Path,
        sync::{
            Arc, RwLock,
            atomic::{AtomicBool, AtomicU64},
        },
    },
    tokio_util::sync::CancellationToken,
};

pub struct Server {
    pub addr: SocketAddr,
    service: JsonRpcService,
    cancel: CancellationToken,
    _tpu_client_runtime: tokio::runtime::Runtime,
}

impl Server {
    pub fn start(
        identity: Arc<Keypair>,
        history: &History,
        blockstore: Arc<Blockstore>,
        ledger_path: &Path,
        genesis_hash: Hash,
        rpc_config: JsonRpcConfig,
    ) -> Self {
        let tip = &history.tip;
        let tip_slot = tip.slot();
        let cluster_info = Arc::new(ClusterInfo::new(
            ContactInfo::new_localhost(&identity.pubkey(), 0),
            identity.clone(),
            SocketAddrSpace::Unspecified,
        ));
        let block_commitment_cache = Arc::new(RwLock::new(BlockCommitmentCache::new(
            HashMap::new(),
            tip.total_epoch_stake(),
            CommitmentSlots::new_from_slot(tip_slot),
        )));
        let optimistically_confirmed_bank =
            OptimisticallyConfirmedBank::locked_from_bank_forks_root(&history.bank_forks);
        let leader_schedule_cache = Arc::new(LeaderScheduleCache::new_from_bank(tip));
        let exit = Arc::new(AtomicBool::new(false));
        // The TPU client is required by the service. Nothing is sent through
        // it, so one thread is enough.
        let tpu_client_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("rpcBenchTpu")
            .enable_all()
            .build()
            .unwrap();
        let cancel = CancellationToken::new();

        let mut attempt = 0;
        loop {
            let port = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
                .and_then(|listener| listener.local_addr())
                .unwrap()
                .port();
            let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
            let tpu_client_socket = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
            let config = JsonRpcServiceConfig {
                rpc_addr: addr,
                rpc_config: rpc_config.clone(),
                snapshot_config: None,
                bank_forks: history.bank_forks.clone(),
                block_commitment_cache: block_commitment_cache.clone(),
                blockstore: blockstore.clone(),
                cluster_info: cluster_info.clone(),
                genesis_hash,
                ledger_path: ledger_path.to_path_buf(),
                validator_exit: create_validator_exit(exit.clone()),
                exit: exit.clone(),
                override_health_check: Arc::new(AtomicBool::new(true)),
                optimistically_confirmed_bank: optimistically_confirmed_bank.clone(),
                highest_finalized: Arc::default(),
                send_transaction_service_config: send_transaction_service::Config::default(),
                max_slots: Arc::new(MaxSlots::default()),
                leader_schedule_cache: leader_schedule_cache.clone(),
                max_complete_transaction_status_slot: Arc::new(AtomicU64::new(tip_slot)),
                max_complete_and_rooted_transaction_status_slot: Arc::new(AtomicU64::new(
                    tip_slot,
                )),
                prioritization_fee_cache: None,
                rpc_tpu_client_args: RpcTpuClientArgs(
                    &identity,
                    tpu_client_socket,
                    tpu_client_runtime.handle().clone(),
                    cancel.clone(),
                ),
            };
            match JsonRpcService::new_with_config(config) {
                Ok(service) => {
                    return Self {
                        addr,
                        service,
                        cancel,
                        _tpu_client_runtime: tpu_client_runtime,
                    };
                }
                // Another process may take the port between the probe and
                // the bind.
                Err(err) if attempt < 10 => {
                    eprintln!("rpc service failed to start on {addr}: {err}, retrying");
                    attempt += 1;
                }
                Err(err) => panic!("rpc service failed to start: {err}"),
            }
        }
    }

    pub fn stop(self) {
        self.cancel.cancel();
        self.service.join().unwrap();
    }
}
