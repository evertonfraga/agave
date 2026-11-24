use std::mem::size_of;

use arbitrary::{Arbitrary, Unstructured};

use solana_sbpf::{program::SBPFVersion, vm::Config};

#[derive(Debug)]
pub struct ConfigTemplate {
    max_call_depth: usize,
    instruction_meter_checkpoint_distance: usize,
    noop_instruction_rate: u32,
    enable_stack_frame_gaps: bool,
    enable_symbol_and_section_labels: bool,
    sanitize_user_provided_values: bool,
    optimize_rodata: bool,
    pub sbpf_version: SBPFVersion,
    aligned_memory_mapping: bool,
}

impl<'a> Arbitrary<'a> for ConfigTemplate {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let bools = u16::arbitrary(u)?;
        // This is how the arbitrary crate quickly generates an enum variant
        let variant = (u64::from(u32::arbitrary(u)?) * 5) >> 32;
        let sbpf_version = match variant {
            0 => SBPFVersion::V0,
            1 => SBPFVersion::V1,
            2 => SBPFVersion::V2,
            3 => SBPFVersion::V3,
            4 => SBPFVersion::V4,
            5 => SBPFVersion::Reserved,
            _ => unreachable!(),
        };
        Ok(ConfigTemplate {
            max_call_depth: usize::from(u8::arbitrary(u)?) + 1, // larger is unreasonable + must be non-zero
            instruction_meter_checkpoint_distance: usize::from(u16::arbitrary(u)?), // larger is unreasonable
            noop_instruction_rate: u32::from(u16::arbitrary(u)?),
            enable_stack_frame_gaps: bools & (1 << 0) != 0,
            enable_symbol_and_section_labels: bools & (1 << 1) != 0,
            sanitize_user_provided_values: bools & (1 << 3) != 0,
            optimize_rodata: bools & (1 << 9) != 0,
            sbpf_version,
            aligned_memory_mapping: bools & (1 << 10) != 0,
        })
    }

    fn size_hint(_: usize) -> (usize, Option<usize>) {
        (
            size_of::<u8>() + size_of::<u16>() + size_of::<f64>() + size_of::<u16>(),
            None,
        )
    }
}

impl From<ConfigTemplate> for Config {
    fn from(template: ConfigTemplate) -> Self {
        let ConfigTemplate {
            max_call_depth,
            instruction_meter_checkpoint_distance,
            noop_instruction_rate,
            enable_stack_frame_gaps,
            enable_symbol_and_section_labels,
            sanitize_user_provided_values,
            optimize_rodata,
            sbpf_version,
            aligned_memory_mapping,
        } = template;
        Config {
            max_call_depth,
            enable_stack_frame_gaps,
            instruction_meter_checkpoint_distance,
            enable_symbol_and_section_labels,
            noop_instruction_rate,
            sanitize_user_provided_values,
            optimize_rodata,
            aligned_memory_mapping: sbpf_version > SBPFVersion::V3 || aligned_memory_mapping,
            ..Default::default()
        }
    }
}

/// What the interpreter did, for [`parrot_diff`] to compare against.
#[allow(dead_code)]
pub struct InterpreterOutcome<'a> {
    pub result: &'a solana_sbpf::error::ProgramResult,
    pub instruction_count: u64,
    pub remaining: u64,
    pub final_pc: u64,
    pub mem: &'a [u8],
    pub stack: &'a [u8],
    pub heap: &'a [u8],
}

/// Run `executable` with the parrot JIT and with the parrot interpreter, on a copy of the initial
/// input `mem` and a budget of `budget`, and panic if either disagrees with the interpreter.
///
/// The result, the instruction count, the remaining budget and the final pc must always match, and
/// the input, stack and heap memory must match when the program succeeded.
#[allow(dead_code)]
#[cfg(all(
    not(target_os = "windows"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub fn parrot_diff(
    executable: &solana_sbpf::elf::Executable<solana_sbpf::test_utils::TestContextObject>,
    mem: &[u8],
    budget: u64,
    interp: &InterpreterOutcome,
) {
    use solana_sbpf::{
        create_vm, ebpf,
        memory_region::MemoryRegion,
        test_utils::TestContextObject,
        vm::{CallFrame, ExecutionMode},
    };
    executable
        .parrot_compile()
        .expect("parrot failed to compile a program that passed verification");
    for (name, mode) in [
        ("parrot jit", ExecutionMode::Jit),
        ("parrot interpreter", ExecutionMode::ParrotInterpreted),
    ] {
        let mut parrot_mem = mem.to_vec();
        let mut context_object = TestContextObject::new(budget);
        let mem_region = MemoryRegion::new(&raw mut parrot_mem[..], ebpf::MM_INPUT_START);
        create_vm!(
            vm,
            executable,
            &mut context_object,
            stack,
            heap,
            vec![mem_region],
            None
        );
        let mut call_frames = vec![CallFrame::default(); executable.get_config().max_call_depth];
        let mut mode = mode;
        let (count, result) = vm.execute_program(executable, &mut mode, &mut call_frames);
        let final_pc = vm.registers[11];
        drop(vm);
        if format!("{:?}", interp.result) != format!("{:?}", result) {
            panic!("[{name}] expected {:?}, but got {:?}", interp.result, result);
        }
        if interp.instruction_count != count {
            panic!(
                "[{name}] expected {} instructions, but got {}",
                interp.instruction_count, count
            );
        }
        if interp.remaining != context_object.remaining {
            panic!(
                "[{name}] expected {} remaining, but got {}",
                interp.remaining, context_object.remaining
            );
        }
        if interp.final_pc != final_pc {
            panic!(
                "[{name}] expected final pc {}, but got {}",
                interp.final_pc, final_pc
            );
        }
        if interp.result.is_ok() {
            if interp.mem != &parrot_mem[..] {
                panic!("[{name}] memory differs: {:?} vs {:?}", interp.mem, parrot_mem);
            }
            if interp.stack != stack.as_slice() {
                panic!("[{}] stack differs", name);
            }
            if interp.heap != heap.as_slice() {
                panic!("[{}] heap differs", name);
            }
        }
    }
}
