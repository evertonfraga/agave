//! Latency percentiles per request type and class, and per-thread CPU time
//! read from /proc/self/task.

use {
    crate::{
        client::Record,
        requests::{CLASSES, REQUEST_TYPES},
    },
    std::{collections::BTreeMap, fs, time::Duration},
};

/// Linux reports thread CPU time in clock ticks of 1/100 s.
const CLOCK_TICKS_PER_SECOND: f64 = 100.0;

#[derive(Default)]
struct Summary {
    latencies_us: Vec<u64>,
    bytes_total: u64,
}

impl Summary {
    fn add(&mut self, record: &Record) {
        self.latencies_us.push(record.latency.as_micros() as u64);
        self.bytes_total += record.bytes as u64;
    }

    fn percentile(&self, percent: usize) -> u64 {
        let index = (self.latencies_us.len() * percent / 100).min(self.latencies_us.len() - 1);
        self.latencies_us[index]
    }

    fn line(&mut self, kind: &str, name: &str, elapsed: Duration) -> String {
        self.latencies_us.sort_unstable();
        let n = self.latencies_us.len();
        let total_us: u64 = self.latencies_us.iter().sum();
        format!(
            "{kind} {name} n={n} rps={:.1} p50_us={} p99_us={} max_us={} mean_us={} \
             bytes_mean={} bytes_total={} errors=0",
            n as f64 / elapsed.as_secs_f64(),
            self.percentile(50),
            self.percentile(99),
            self.latencies_us[n - 1],
            total_us / n as u64,
            self.bytes_total / n as u64,
            self.bytes_total,
        )
    }
}

/// Prints the RPCSTATS and RPCTYPE lines and the per-class percentile phases.
pub fn report(records: &[Record], elapsed: Duration) {
    let mut types: Vec<Summary> = REQUEST_TYPES.iter().map(|_| Summary::default()).collect();
    let mut classes: Vec<Summary> = CLASSES.iter().map(|_| Summary::default()).collect();
    for record in records {
        types[record.request_type].add(record);
        let class = REQUEST_TYPES[record.request_type].class;
        classes[CLASSES.iter().position(|c| *c == class).unwrap()].add(record);
    }
    for (class, summary) in CLASSES.iter().zip(&mut classes) {
        if summary.latencies_us.is_empty() {
            continue;
        }
        println!("{}", summary.line("RPCSTATS", class.name(), elapsed));
        println!(
            "BENCH_PHASE:{}_p50:{:.6}",
            class.name(),
            summary.percentile(50) as f64 / 1e6
        );
        println!(
            "BENCH_PHASE:{}_p99:{:.6}",
            class.name(),
            summary.percentile(99) as f64 / 1e6
        );
    }
    for (request_type, summary) in REQUEST_TYPES.iter().zip(&mut types) {
        if !summary.latencies_us.is_empty() {
            println!("{}", summary.line("RPCTYPE", request_type.name, elapsed));
        }
    }
}

/// CPU ticks used so far by each live thread, keyed by thread id, with the
/// thread name. Empty where /proc is not available.
pub fn thread_cpu() -> BTreeMap<u32, (String, u64)> {
    let mut threads = BTreeMap::new();
    let Ok(tasks) = fs::read_dir("/proc/self/task") else {
        return threads;
    };
    for task in tasks.flatten() {
        let Some(tid) = task.file_name().to_str().and_then(|s| s.parse().ok()) else {
            continue;
        };
        let Ok(stat) = fs::read_to_string(task.path().join("stat")) else {
            continue;
        };
        // The name is in parentheses and can contain spaces, so the other
        // fields are counted from the last ')'. utime and stime are fields 14
        // and 15 of the line, which are 12 and 13 after the name.
        let (Some(open), Some(close)) = (stat.find('('), stat.rfind(')')) else {
            continue;
        };
        let name = stat[open + 1..close].to_string();
        let fields: Vec<&str> = stat[close + 1..].split_whitespace().collect();
        let ticks = |i: usize| fields.get(i).and_then(|f| f.parse::<u64>().ok()).unwrap_or(0);
        threads.insert(tid, (name, ticks(11) + ticks(12)));
    }
    threads
}

/// Prints server and client CPU seconds over a measured window.
///
/// Server runtime threads are named solRpcEl. The ones in `workers`, taken
/// right after the service started, are the worker threads; any other is a
/// blocking-pool thread. Threads that exited inside the window are missing
/// from `end`, so their CPU time is lost; their number is printed.
pub fn report_cpu(
    workers: &BTreeMap<u32, (String, u64)>,
    start: &BTreeMap<u32, (String, u64)>,
    end: &BTreeMap<u32, (String, u64)>,
    elapsed: Duration,
    client_cpus: usize,
) {
    if end.is_empty() {
        return;
    }
    let mut worker = 0;
    let mut blocking = 0;
    let mut client = 0;
    for (tid, (name, ticks)) in end {
        let used = ticks.saturating_sub(start.get(tid).map_or(0, |(_, ticks)| *ticks));
        if name == "solRpcEl" {
            if workers.contains_key(tid) {
                worker += used;
            } else {
                blocking += used;
            }
        } else if name == "rpcBenchClient" {
            client += used;
        }
    }
    let lost = start
        .iter()
        .filter(|(tid, (name, _))| name == "solRpcEl" && !end.contains_key(tid))
        .count();
    let seconds = |ticks: u64| ticks as f64 / CLOCK_TICKS_PER_SECOND;
    println!("BENCH_PHASE:server_worker_cpu:{:.6}", seconds(worker));
    println!("BENCH_PHASE:server_blocking_cpu:{:.6}", seconds(blocking));
    println!("BENCH_PHASE:client_cpu:{:.6}", seconds(client));
    if lost > 0 {
        println!("server threads that exited during the run: {lost}");
    }
    let client_share = seconds(client) / (elapsed.as_secs_f64() * client_cpus as f64);
    if client_share > 0.7 {
        println!(
            "WARNING: client threads used {:.0}% of their {client_cpus} CPUs; the result may be \
             bound by the client",
            client_share * 100.0
        );
    }
}
