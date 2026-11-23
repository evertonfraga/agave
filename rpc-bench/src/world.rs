//! Seeded synthetic accounts and the transaction builders shared by the
//! history and the simulateTransaction requests.

use {
    rand_chacha::{
        ChaCha8Rng,
        rand_core::{RngCore, SeedableRng},
    },
    solana_account::Account,
    solana_compute_budget_interface::ComputeBudgetInstruction,
    solana_hash::Hash,
    solana_instruction::Instruction,
    solana_keypair::Keypair,
    solana_program_pack::Pack,
    solana_pubkey::Pubkey,
    solana_rent::Rent,
    solana_signer::Signer,
    solana_transaction::Transaction,
    spl_token_interface::state::{Account as TokenAccount, AccountState, Mint},
    std::collections::BTreeMap,
};

pub const TOKEN_DECIMALS: u8 = 6;
pub const TOKEN_BALANCE: u64 = 1_000_000_000_000;
/// More than any account can hold, so a transfer of it always fails.
pub const FAILING_AMOUNT: u64 = 1_000_000_000_000_000;
pub const MAX_TRANSACTION_SIZE: usize = 1232;
pub const SCAN_ACCOUNT_SIZE: usize = 256;
pub const POOL_ACCOUNT_SIZE: usize = 1024;
pub const SCAN_POSITION: [u8; 8] = *b"position";
pub const SCAN_MARKET: [u8; 8] = *b"market\0\0";
pub const NUM_MARKETS: usize = 100;
const COMPUTE_UNIT_PRICE: u64 = 1_000;

pub fn new_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Returns a value in `lo..hi`.
pub fn range(rng: &mut ChaCha8Rng, lo: u64, hi: u64) -> u64 {
    lo + rng.next_u64() % (hi - lo)
}

pub fn index(rng: &mut ChaCha8Rng, len: usize) -> usize {
    (rng.next_u64() % len as u64) as usize
}

pub fn new_pubkey(rng: &mut ChaCha8Rng) -> Pubkey {
    let mut bytes = [0u8; 32];
    rng.fill_bytes(&mut bytes);
    Pubkey::new_from_array(bytes)
}

fn new_keypair(rng: &mut ChaCha8Rng) -> Keypair {
    let mut bytes = [0u8; 32];
    rng.fill_bytes(&mut bytes);
    Keypair::new_from_array(bytes)
}

/// Zipf sampler with exponent 1, using integer weights only, so that the
/// draws are the same on every architecture.
pub struct Zipf {
    cumulative: Vec<u64>,
}

impl Zipf {
    pub fn new(len: usize) -> Self {
        let mut total = 0u64;
        let cumulative = (1..=len as u64)
            .map(|rank| {
                total += 1_000_000_000 / rank;
                total
            })
            .collect();
        Self { cumulative }
    }

    pub fn sample(&self, rng: &mut ChaCha8Rng) -> usize {
        let x = rng.next_u64() % self.cumulative.last().unwrap();
        self.cumulative.partition_point(|&c| c <= x)
    }
}

pub struct Sizes {
    pub wallets: usize,
    pub active_owners: usize,
    pub token_mints: usize,
    pub token_accounts: usize,
    pub whales: usize,
    pub whale_accounts: usize,
    pub t22_mints: usize,
    pub scan_accounts: usize,
    pub pool_accounts: usize,
}

impl Sizes {
    /// `divisor` 1 is the full world, 20 is the small world used for checks.
    pub fn new(divisor: usize) -> Self {
        Self {
            wallets: 250_000 / divisor,
            active_owners: 10_000 / divisor,
            token_mints: (500 / divisor).max(20),
            token_accounts: 300_000 / divisor,
            whales: 4,
            whale_accounts: 2_000 / divisor,
            t22_mints: 20,
            scan_accounts: 20_000 / divisor,
            pool_accounts: 5_000 / divisor,
        }
    }
}

pub struct TokenHolding {
    pub address: Pubkey,
    pub mint: usize,
    pub owner: Pubkey,
}

pub struct World {
    pub wallets: Vec<Pubkey>,
    pub wallet_lamports: Vec<u64>,
    /// Signers of the first `active.len()` wallets.
    pub active: Vec<Keypair>,
    pub whales: Vec<Pubkey>,
    pub token_mints: Vec<Pubkey>,
    pub token_accounts: Vec<TokenHolding>,
    pub token_mint_holders: Vec<Vec<usize>>,
    pub t22_mints: Vec<Pubkey>,
    pub t22_accounts: Vec<TokenHolding>,
    pub t22_mint_holders: Vec<Vec<usize>>,
    /// Token and Token-2022 account indexes owned by each active wallet.
    pub owner_token_accounts: Vec<Vec<usize>>,
    pub owner_t22_accounts: Vec<Vec<usize>>,
    pub scan_program: Pubkey,
    pub markets: Vec<Pubkey>,
    pub scan_accounts: Vec<Pubkey>,
    /// Number of POSITION accounts per market.
    pub scan_positions: Vec<usize>,
    pub pool_program: Pubkey,
    pub pool_accounts: Vec<Pubkey>,
    pub pool_data: Vec<Vec<u8>>,
}

pub fn token_account_data(mint: &Pubkey, owner: &Pubkey, amount: u64) -> Vec<u8> {
    // Token-2022 accounts without extensions use the same 165-byte layout.
    let state = TokenAccount {
        mint: *mint,
        owner: *owner,
        amount,
        state: AccountState::Initialized,
        ..TokenAccount::default()
    };
    let mut data = vec![0; TokenAccount::LEN];
    state.pack_into_slice(&mut data);
    data
}

fn mint_data(supply: u64) -> Vec<u8> {
    let state = Mint {
        supply,
        decimals: TOKEN_DECIMALS,
        is_initialized: true,
        ..Mint::default()
    };
    let mut data = vec![0; Mint::LEN];
    state.pack_into_slice(&mut data);
    data
}

impl World {
    pub fn new(rng: &mut ChaCha8Rng, sizes: &Sizes) -> Self {
        let active: Vec<_> = (0..sizes.active_owners)
            .map(|_| new_keypair(rng))
            .collect();
        let wallets: Vec<_> = active
            .iter()
            .map(|keypair| keypair.pubkey())
            .chain((sizes.active_owners..sizes.wallets).map(|_| new_pubkey(rng)))
            .collect();
        let wallet_lamports = wallets
            .iter()
            .map(|_| range(rng, 100_000_000, 100_000_000_000))
            .collect();
        let whales: Vec<_> = (0..sizes.whales).map(|_| new_pubkey(rng)).collect();

        let token_mints: Vec<_> = (0..sizes.token_mints).map(|_| new_pubkey(rng)).collect();
        let zipf = Zipf::new(sizes.token_mints);
        let mut token_accounts = Vec::with_capacity(sizes.token_accounts);
        let mut owner_token_accounts = vec![vec![]; sizes.active_owners];
        for whale in &whales {
            for _ in 0..sizes.whale_accounts {
                let mint = zipf.sample(rng);
                token_accounts.push(TokenHolding {
                    address: new_pubkey(rng),
                    mint,
                    owner: *whale,
                });
            }
        }
        for (owner, accounts) in owner_token_accounts.iter_mut().enumerate() {
            for _ in 0..range(rng, 5, 13) {
                accounts.push(token_accounts.len());
                let mint = zipf.sample(rng);
                token_accounts.push(TokenHolding {
                    address: new_pubkey(rng),
                    mint,
                    owner: wallets[owner],
                });
            }
        }
        while token_accounts.len() < sizes.token_accounts {
            // Single accounts belong to wallets that never sign, so the
            // balances of those wallets stay at their genesis values.
            let owner = wallets[range(rng, sizes.active_owners as u64, sizes.wallets as u64) as usize];
            let mint = zipf.sample(rng);
            token_accounts.push(TokenHolding {
                address: new_pubkey(rng),
                mint,
                owner,
            });
        }
        let token_mint_holders = holders(&token_accounts, sizes.token_mints);

        let t22_mints: Vec<_> = (0..sizes.t22_mints).map(|_| new_pubkey(rng)).collect();
        let mut t22_accounts = Vec::with_capacity(2 * sizes.active_owners);
        let mut owner_t22_accounts = vec![vec![]; sizes.active_owners];
        for (owner, accounts) in owner_t22_accounts.iter_mut().enumerate() {
            for _ in 0..2 {
                accounts.push(t22_accounts.len());
                t22_accounts.push(TokenHolding {
                    address: new_pubkey(rng),
                    mint: index(rng, sizes.t22_mints),
                    owner: wallets[owner],
                });
            }
        }
        let t22_mint_holders = holders(&t22_accounts, sizes.t22_mints);

        let scan_program = new_pubkey(rng);
        let markets: Vec<_> = (0..NUM_MARKETS).map(|_| new_pubkey(rng)).collect();
        let scan_accounts = (0..sizes.scan_accounts).map(|_| new_pubkey(rng)).collect();
        let pool_program = new_pubkey(rng);
        let pool_accounts = (0..sizes.pool_accounts).map(|_| new_pubkey(rng)).collect();
        let pool_data = (0..sizes.pool_accounts)
            .map(|_| {
                let mut data = vec![0; POOL_ACCOUNT_SIZE];
                rng.fill_bytes(&mut data);
                data
            })
            .collect();

        Self {
            wallets,
            wallet_lamports,
            active,
            whales,
            token_mints,
            token_accounts,
            token_mint_holders,
            t22_mints,
            t22_accounts,
            t22_mint_holders,
            owner_token_accounts,
            owner_t22_accounts,
            scan_program,
            markets,
            scan_accounts,
            scan_positions: vec![0; NUM_MARKETS],
            pool_program,
            pool_accounts,
            pool_data,
        }
    }

    /// Adds every world account to `accounts`. The scan account data is drawn
    /// here, and the POSITION count per market is recorded for the checks.
    pub fn add_genesis_accounts(
        &mut self,
        rng: &mut ChaCha8Rng,
        rent: &Rent,
        accounts: &mut BTreeMap<Pubkey, Account>,
    ) {
        for (wallet, lamports) in self.wallets.iter().zip(&self.wallet_lamports) {
            accounts.insert(
                *wallet,
                owned(*lamports, vec![], &solana_sdk_ids::system_program::id()),
            );
        }
        let token_lamports = rent.minimum_balance(TokenAccount::LEN);
        let mint_lamports = rent.minimum_balance(Mint::LEN);
        for (program, mints, holdings, mint_holders) in [
            (
                spl_token_interface::id(),
                &self.token_mints,
                &self.token_accounts,
                &self.token_mint_holders,
            ),
            (
                spl_token_2022_interface::id(),
                &self.t22_mints,
                &self.t22_accounts,
                &self.t22_mint_holders,
            ),
        ] {
            for (mint, holders) in mints.iter().zip(mint_holders) {
                let supply = TOKEN_BALANCE * holders.len() as u64;
                accounts.insert(*mint, owned(mint_lamports, mint_data(supply), &program));
            }
            for holding in holdings {
                let data = token_account_data(&mints[holding.mint], &holding.owner, TOKEN_BALANCE);
                accounts.insert(holding.address, owned(token_lamports, data, &program));
            }
        }

        let scan_lamports = rent.minimum_balance(SCAN_ACCOUNT_SIZE);
        for address in &self.scan_accounts {
            let mut data = vec![0; SCAN_ACCOUNT_SIZE];
            rng.fill_bytes(&mut data);
            let market = index(rng, NUM_MARKETS);
            if range(rng, 0, 10) == 0 {
                data[..8].copy_from_slice(&SCAN_MARKET);
            } else {
                data[..8].copy_from_slice(&SCAN_POSITION);
                self.scan_positions[market] += 1;
            }
            data[8..40].copy_from_slice(self.markets[market].as_ref());
            accounts.insert(*address, owned(scan_lamports, data, &self.scan_program));
        }
        let pool_lamports = rent.minimum_balance(POOL_ACCOUNT_SIZE);
        for (address, data) in self.pool_accounts.iter().zip(&self.pool_data) {
            accounts.insert(
                *address,
                owned(pool_lamports, data.clone(), &self.pool_program),
            );
        }
    }

    pub fn num_accounts(&self) -> usize {
        self.wallets.len()
            + self.token_mints.len()
            + self.token_accounts.len()
            + self.t22_mints.len()
            + self.t22_accounts.len()
            + self.scan_accounts.len()
            + self.pool_accounts.len()
    }

    /// A random Token account of the same mint as `holding`, other than it.
    fn token_destination(&self, rng: &mut ChaCha8Rng, holding: usize) -> &Pubkey {
        let holders = &self.token_mint_holders[self.token_accounts[holding].mint];
        other_holder(rng, holders, holding, &self.token_accounts)
    }

    fn t22_destination(&self, rng: &mut ChaCha8Rng, holding: usize) -> &Pubkey {
        let holders = &self.t22_mint_holders[self.t22_accounts[holding].mint];
        other_holder(rng, holders, holding, &self.t22_accounts)
    }

    fn p_token_transfer(&self, rng: &mut ChaCha8Rng, owner: usize, n: usize) -> Instruction {
        let accounts = &self.owner_token_accounts[owner];
        let source = accounts[n % accounts.len()];
        spl_token_interface::instruction::transfer(
            &spl_token_interface::id(),
            &self.token_accounts[source].address,
            self.token_destination(rng, source),
            &self.wallets[owner],
            &[],
            range(rng, 1, 1_000),
        )
        .unwrap()
    }

    fn t22_transfer(&self, rng: &mut ChaCha8Rng, owner: usize, n: usize, amount: u64) -> Instruction {
        let accounts = &self.owner_t22_accounts[owner];
        let source = accounts[n % accounts.len()];
        let holding = &self.t22_accounts[source];
        spl_token_2022_interface::instruction::transfer_checked(
            &spl_token_2022_interface::id(),
            &holding.address,
            &self.t22_mints[holding.mint],
            self.t22_destination(rng, source),
            &self.wallets[owner],
            &[],
            amount,
            TOKEN_DECIMALS,
        )
        .unwrap()
    }

    fn memo(rng: &mut ChaCha8Rng, len: usize) -> Instruction {
        let memo: Vec<u8> = (0..len).map(|_| b'a' + (rng.next_u32() % 26) as u8).collect();
        spl_memo_interface::instruction::build_memo(&spl_memo_interface::v3::id(), &memo, &[])
    }

    fn sign(&self, owner: usize, instructions: &[Instruction], blockhash: Hash) -> Transaction {
        let signer = &self.active[owner];
        let transaction = Transaction::new_signed_with_payer(
            instructions,
            Some(&signer.pubkey()),
            &[signer],
            blockhash,
        );
        let size = wincode::serialize(&transaction).unwrap().len();
        assert!(size <= MAX_TRANSACTION_SIZE, "transaction size {size}");
        transaction
    }

    /// Builds one transaction of the given kind, signed by active wallet `owner`.
    pub fn transaction(
        &self,
        rng: &mut ChaCha8Rng,
        kind: TxKind,
        owner: usize,
        blockhash: Hash,
    ) -> Transaction {
        let compute_budget = |units| {
            [
                ComputeBudgetInstruction::set_compute_unit_limit(units),
                ComputeBudgetInstruction::set_compute_unit_price(COMPUTE_UNIT_PRICE),
            ]
        };
        let mut instructions = vec![];
        match kind {
            TxKind::TokenTransfer => {
                instructions.extend(compute_budget(10_000));
                instructions.push(self.p_token_transfer(rng, owner, 0));
            }
            TxKind::SystemTransfer => {
                let to = &self.wallets[index(rng, self.active.len())];
                let signer = &self.active[owner];
                return solana_system_transaction::transfer(signer, to, 1_000_000, blockhash);
            }
            TxKind::T22TransferMemo => {
                let amount = range(rng, 1, 1_000);
                instructions.push(self.t22_transfer(rng, owner, 0, amount));
                instructions.push(Self::memo(rng, 64));
            }
            TxKind::CreateAta(count) => {
                for _ in 0..count {
                    let mint = &self.t22_mints[index(rng, self.t22_mints.len())];
                    instructions.push(
                        spl_associated_token_account_interface::instruction::create_associated_token_account_idempotent(
                            &self.wallets[owner],
                            &new_pubkey(rng),
                            mint,
                            &spl_token_2022_interface::id(),
                        ),
                    );
                }
            }
            TxKind::Swap { fail } => {
                instructions.extend(compute_budget(400_000));
                for n in 0..4 {
                    instructions.push(self.p_token_transfer(rng, owner, n));
                }
                for n in 0..4 {
                    let amount = if fail && n == 3 {
                        FAILING_AMOUNT
                    } else {
                        range(rng, 1, 1_000)
                    };
                    instructions.push(self.t22_transfer(rng, owner, n, amount));
                }
                instructions.push(Self::memo(rng, 32));
            }
            TxKind::Batch { transfers, fail } => {
                instructions.extend(compute_budget(1_400_000));
                // One source, destination, mint and authority for every
                // transfer, so that many transfers fit in one transaction.
                let source = self.owner_t22_accounts[owner][0];
                let holding = &self.t22_accounts[source];
                let destination = self.t22_destination(rng, source);
                for n in 0..transfers {
                    let amount = if fail && n + 1 == transfers {
                        FAILING_AMOUNT
                    } else {
                        1
                    };
                    instructions.push(
                        spl_token_2022_interface::instruction::transfer_checked(
                            &spl_token_2022_interface::id(),
                            &holding.address,
                            &self.t22_mints[holding.mint],
                            destination,
                            &self.wallets[owner],
                            &[],
                            amount,
                            TOKEN_DECIMALS,
                        )
                        .unwrap(),
                    );
                }
            }
        }
        self.sign(owner, &instructions, blockhash)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxKind {
    TokenTransfer,
    SystemTransfer,
    T22TransferMemo,
    CreateAta(usize),
    Swap { fail: bool },
    Batch { transfers: usize, fail: bool },
}

fn owned(lamports: u64, data: Vec<u8>, owner: &Pubkey) -> Account {
    Account {
        lamports,
        data,
        owner: *owner,
        executable: false,
        rent_epoch: 0,
    }
}

fn holders(holdings: &[TokenHolding], num_mints: usize) -> Vec<Vec<usize>> {
    let mut holders = vec![vec![]; num_mints];
    for (i, holding) in holdings.iter().enumerate() {
        holders[holding.mint].push(i);
    }
    for (mint, holders) in holders.iter().enumerate() {
        assert!(holders.len() >= 2, "mint {mint} has fewer than two holders");
    }
    holders
}

fn other_holder<'a>(
    rng: &mut ChaCha8Rng,
    holders: &[usize],
    holding: usize,
    holdings: &'a [TokenHolding],
) -> &'a Pubkey {
    let mut pick = index(rng, holders.len());
    if holders[pick] == holding {
        pick = (pick + 1) % holders.len();
    }
    &holdings[holders[pick]].address
}
