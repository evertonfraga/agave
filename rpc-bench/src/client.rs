//! Closed-loop HTTP/1.1 keep-alive client. Each connection has one request
//! in flight and takes the next index of the shared sequence.

use {
    crate::requests::REQUEST_TYPES,
    http_body_util::{BodyExt, Full},
    hyper::{
        Request, StatusCode,
        body::Bytes,
        client::conn::http1::{self, SendRequest},
        header::{CONTENT_TYPE, HOST, HeaderValue},
    },
    hyper_util::rt::TokioIo,
    std::{
        net::SocketAddr,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::{Duration, Instant},
    },
    tokio::{net::TcpStream, runtime::Runtime, sync::Barrier},
};

/// A successful JSON-RPC response from jsonrpc-core starts with this prefix.
/// Errors put an `error` member in its place.
const SUCCESS_PREFIX: &[u8] = br#"{"jsonrpc":"2.0","result""#;

pub struct Record {
    pub request_type: usize,
    pub latency: Duration,
    pub bytes: usize,
}

pub struct Outcome {
    pub elapsed: Duration,
    pub records: Vec<Record>,
}

pub fn new_runtime(threads: usize) -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads)
        .thread_name("rpcBenchClient")
        .enable_all()
        .build()
        .unwrap()
}

async fn connect(addr: SocketAddr) -> Result<SendRequest<Full<Bytes>>, String> {
    let stream = TcpStream::connect(addr)
        .await
        .map_err(|err| format!("connect to {addr}: {err}"))?;
    stream.set_nodelay(true).unwrap();
    let (sender, connection) = http1::handshake(TokioIo::new(stream))
        .await
        .map_err(|err| format!("handshake with {addr}: {err}"))?;
    tokio::spawn(connection);
    Ok(sender)
}

/// Sends one request and returns the response body, or the error text.
pub async fn send(
    sender: &mut SendRequest<Full<Bytes>>,
    host: &HeaderValue,
    body: Bytes,
) -> Result<Bytes, String> {
    let request = Request::post("/")
        .header(HOST, host)
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(body))
        .unwrap();
    sender.ready().await.map_err(|err| err.to_string())?;
    let response = sender
        .send_request(request)
        .await
        .map_err(|err| err.to_string())?;
    let status = response.status();
    let body = response
        .into_body()
        .collect()
        .await
        .map_err(|err| err.to_string())?
        .to_bytes();
    if status != StatusCode::OK || !body.starts_with(SUCCESS_PREFIX) {
        let len = body.len().min(2_000);
        return Err(format!(
            "status {status}: {}",
            String::from_utf8_lossy(&body[..len])
        ));
    }
    Ok(body)
}

async fn connection_loop(
    mut sender: SendRequest<Full<Bytes>>,
    host: HeaderValue,
    pools: Arc<Vec<Vec<Bytes>>>,
    sequence: Arc<Vec<(usize, usize)>>,
    next: Arc<AtomicUsize>,
    start: Arc<Barrier>,
) -> Result<Vec<Record>, String> {
    start.wait().await;
    let mut records = Vec::with_capacity(sequence.len() / 16);
    loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        let Some(&(request_type, body)) = sequence.get(i) else {
            return Ok(records);
        };
        let sent = Instant::now();
        let response = send(&mut sender, &host, pools[request_type][body].clone())
            .await
            .map_err(|err| format!("{}: {err}", REQUEST_TYPES[request_type].name))?;
        records.push(Record {
            request_type,
            latency: sent.elapsed(),
            bytes: response.len(),
        });
    }
}

/// Opens `concurrency` connections, then sends every request of `sequence`
/// and returns the wall time from the start barrier to the last response.
pub fn run(
    runtime: &Runtime,
    addr: SocketAddr,
    pools: Arc<Vec<Vec<Bytes>>>,
    sequence: Arc<Vec<(usize, usize)>>,
    concurrency: usize,
) -> Result<Outcome, String> {
    runtime.block_on(async move {
        let host = HeaderValue::from_str(&addr.to_string()).unwrap();
        let next = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(concurrency + 1));
        let mut tasks = Vec::with_capacity(concurrency);
        for _ in 0..concurrency {
            let sender = connect(addr).await?;
            tasks.push(tokio::spawn(connection_loop(
                sender,
                host.clone(),
                pools.clone(),
                sequence.clone(),
                next.clone(),
                start.clone(),
            )));
        }
        start.wait().await;
        let started = Instant::now();
        let mut records = Vec::with_capacity(sequence.len());
        let mut first_error = None;
        for task in tasks {
            match task.await.unwrap() {
                Ok(task_records) => records.extend(task_records),
                Err(err) => {
                    // Stop the other connections at the next request.
                    next.store(sequence.len(), Ordering::Relaxed);
                    first_error.get_or_insert(err);
                }
            }
        }
        let elapsed = started.elapsed();
        match first_error {
            Some(err) => Err(err),
            None => Ok(Outcome { elapsed, records }),
        }
    })
}

/// A single connection for sequential requests, used by the checks.
pub struct Connection {
    sender: SendRequest<Full<Bytes>>,
    host: HeaderValue,
}

impl Connection {
    pub fn open(runtime: &Runtime, addr: SocketAddr) -> Result<Self, String> {
        let sender = runtime.block_on(connect(addr))?;
        Ok(Self {
            sender,
            host: HeaderValue::from_str(&addr.to_string()).unwrap(),
        })
    }

    pub fn request(&mut self, runtime: &Runtime, body: Bytes) -> Result<serde_json::Value, String> {
        let response = runtime.block_on(send(&mut self.sender, &self.host, body))?;
        let mut value: serde_json::Value =
            serde_json::from_slice(&response).map_err(|err| err.to_string())?;
        Ok(value["result"].take())
    }
}
