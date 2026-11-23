//! Sends each request type once or a few times, one at a time, and checks
//! the responses against the world and history that were generated.

use {
    crate::{
        client::Connection,
        history::{HISTORY_MIX, History},
        requests::{self, REQUEST_TYPES},
        world::{TxKind, World, index},
    },
    base64::{Engine, prelude::BASE64_STANDARD},
    rand_chacha::ChaCha8Rng,
    serde_json::Value,
    solana_pubkey::Pubkey,
    std::net::SocketAddr,
    tokio::runtime::Runtime,
};

const FEE_PER_SIGNATURE: u64 = 5_000;

fn check(ok: bool, what: impl FnOnce() -> String) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what()) }
}

fn len(value: &Value) -> usize {
    value.as_array().map_or(0, Vec::len)
}

pub fn run(
    rng: &mut ChaCha8Rng,
    runtime: &Runtime,
    addr: SocketAddr,
    world: &World,
    history: &History,
    transactions_per_slot: usize,
) -> Result<(), String> {
    let mut connection = Connection::open(runtime, addr)?;
    let mut request = |body| connection.request(runtime, body);
    let tip = &history.tip;

    let expected_failures = history
        .transactions
        .iter()
        .filter(|t| t.kind == TxKind::Swap { fail: true })
        .count() as u64;
    check(
        history.executed == history.transactions.len() as u64 && history.failed == expected_failures,
        || {
            format!(
                "history executed {} and failed {} transactions, expected {} and {expected_failures}",
                history.executed,
                history.failed,
                history.transactions.len()
            )
        },
    )?;

    let slot = request(requests::body("getSlot", Value::Array(vec![])))?;
    check(slot == tip.slot(), || format!("getSlot returned {slot}"))?;
    let blockhash = request(requests::body("getLatestBlockhash", Value::Array(vec![])))?;
    check(
        blockhash["value"]["blockhash"] == tip.last_blockhash().to_string(),
        || format!("getLatestBlockhash returned {blockhash}"),
    )?;

    for _ in 0..16 {
        let wallet = world.active.len() + index(rng, world.wallets.len() - world.active.len());
        let balance = request(requests::get_balance(&world.wallets[wallet]))?;
        check(balance["value"] == world.wallet_lamports[wallet], || {
            format!("getBalance of wallet {wallet} returned {balance}")
        })?;
    }
    for _ in 0..16 {
        let pool = index(rng, world.pool_accounts.len());
        let info = request(requests::get_account_info(&world.pool_accounts[pool]))?;
        let data = info["value"]["data"][0].as_str().unwrap_or_default();
        check(
            BASE64_STANDARD.decode(data).ok().as_ref() == Some(&world.pool_data[pool])
                && info["value"]["owner"] == world.pool_program.to_string(),
            || format!("getAccountInfo of pool account {pool} returned other data"),
        )?;
    }
    let addresses: Vec<Pubkey> = (0..100)
        .map(|i| world.token_accounts[i * 7 % world.token_accounts.len()].address)
        .collect();
    let accounts = request(requests::get_multiple_accounts(&addresses))?;
    let values = accounts["value"].as_array().cloned().unwrap_or_default();
    check(
        values.len() == 100 && values.iter().all(|v| !v.is_null()),
        || "getMultipleAccounts did not return 100 accounts".to_string(),
    )?;

    let known: Vec<_> = (0..48)
        .map(|_| &history.transactions[index(rng, history.transactions.len())])
        .collect();
    let mut signatures: Vec<_> = known.iter().map(|t| t.signature).collect();
    signatures.extend((0..16).map(|i| solana_signature::Signature::from([i as u8 + 1; 64])));
    let statuses = request(requests::get_signature_statuses(&signatures))?;
    for (i, status) in statuses["value"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        match known.get(i) {
            Some(transaction) => check(
                status["slot"] == transaction.slot
                    && status["err"].is_null() != (transaction.kind == TxKind::Swap { fail: true }),
                || format!("signature status {i} is {status}"),
            )?,
            None => check(status.is_null(), || {
                format!("unknown signature {i} has status {status}")
            })?,
        }
    }
    check(len(&statuses["value"]) == 64, || {
        "getSignatureStatuses did not return 64 statuses".to_string()
    })?;

    for (kind, _) in HISTORY_MIX {
        let transaction = history
            .transactions
            .iter()
            .find(|t| t.kind == kind)
            .ok_or_else(|| format!("no history transaction of kind {kind:?}"))?;
        let result = request(requests::get_transaction(&transaction.signature))?;
        let meta = &result["meta"];
        let failed = kind == TxKind::Swap { fail: true };
        check(
            meta["err"].is_null() != failed && len(&meta["logMessages"]) > 0,
            || format!("getTransaction for {kind:?} returned meta {meta}"),
        )?;
        if matches!(kind, TxKind::CreateAta(_)) {
            check(len(&meta["innerInstructions"]) > 0, || {
                format!("getTransaction for {kind:?} has no inner instructions")
            })?;
        }
        if matches!(kind, TxKind::SystemTransfer | TxKind::T22TransferMemo) {
            check(meta["fee"] == FEE_PER_SIGNATURE, || {
                format!("fee of {kind:?} is {}", meta["fee"])
            })?;
        }
        println!(
            "history kind={kind:?} units={}",
            meta["computeUnitsConsumed"]
        );
    }

    for owner in [0, world.active.len() / 2, world.active.len() - 1] {
        let accounts = request(requests::get_token_accounts_by_owner(&world.wallets[owner]))?;
        check(
            len(&accounts["value"]) == world.owner_token_accounts[owner].len(),
            || format!("getTokenAccountsByOwner of owner {owner} returned {}", len(&accounts["value"])),
        )?;
        let signatures = request(requests::get_signatures_for_address(&world.wallets[owner], 20))?;
        check(
            len(&signatures) == history.wallet_transactions[owner].min(20),
            || format!("getSignaturesForAddress of owner {owner} returned {}", len(&signatures)),
        )?;
    }

    for market in [0, world.markets.len() - 1] {
        let accounts = request(requests::get_program_accounts(
            &world.scan_program,
            requests::scan_filters(world, market),
        ))?;
        check(len(&accounts) == world.scan_positions[market], || {
            format!("getProgramAccounts for market {market} returned {}", len(&accounts))
        })?;
    }
    let mint = world.token_mints.len() / 2;
    let accounts = request(requests::get_program_accounts(
        &spl_token_interface::id(),
        requests::mint_filters(&world.token_mints[mint]),
    ))?;
    check(len(&accounts) == world.token_mint_holders[mint].len(), || {
        format!("getProgramAccounts for mint {mint} returned {}", len(&accounts))
    })?;
    let whale_accounts = request(requests::get_token_accounts_by_owner(&world.whales[0]))?;
    let expected = world
        .token_accounts
        .iter()
        .filter(|a| a.owner == world.whales[0])
        .count();
    check(len(&whale_accounts["value"]) == expected, || {
        format!("whale has {} token accounts, expected {expected}", len(&whale_accounts["value"]))
    })?;

    let slot = *history.transaction_slots.last().unwrap();
    let block = request(requests::get_block(slot, "full"))?;
    check(len(&block["transactions"]) == transactions_per_slot, || {
        format!("getBlock {slot} returned {} transactions", len(&block["transactions"]))
    })?;
    let block = request(requests::get_block(slot, "signatures"))?;
    check(len(&block["signatures"]) == transactions_per_slot, || {
        format!("getBlock {slot} returned {} signatures", len(&block["signatures"]))
    })?;
    let signatures = request(requests::get_signatures_for_address(
        &spl_token_interface::id(),
        1000,
    ))?;
    check(
        len(&signatures) == history.token_program_transactions.min(1000),
        || format!("getSignaturesForAddress for the Token program returned {}", len(&signatures)),
    )?;

    let blockhash = tip.last_blockhash();
    for request_type in REQUEST_TYPES.iter().filter(|t| t.name.starts_with("simulate.")) {
        for n in 0..2 {
            let body = requests::exec_body(rng, world, request_type.name, n, blockhash);
            let result = request(body)?;
            let value = &result["value"];
            let should_fail = request_type.name == "simulate.fail";
            check(
                value["err"].is_null() != should_fail && value["unitsConsumed"].as_u64() > Some(0),
                || format!("{} returned {value}", request_type.name),
            )?;
            if request_type.name == "simulate.cpi" {
                check(len(&value["innerInstructions"]) > 0, || {
                    format!("{} returned no inner instructions", request_type.name)
                })?;
            }
            if n == 0 {
                println!("exec type={} units={}", request_type.name, value["unitsConsumed"]);
            }
        }
    }
    Ok(())
}
