//! Build script for nano-vllm-rs.
//!
//! Compiles custom CUDA kernels when the `cuda` feature is enabled.

fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    use std::path::PathBuf;

    println!("cargo:rerun-if-changed=kernels/");

    let cuda_files = ["kernels/flash_attn_fwd.cu"];

    // Check if nvcc is available
    let nvcc_check = std::process::Command::new("nvcc").arg("--version").output();

    if nvcc_check.is_err() {
        println!("cargo:warning=nvcc not found, skipping CUDA kernel compilation");
        return;
    }

    // Determine CUDA include path
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include = PathBuf::from(&cuda_path).join("include");

    // Build CUDA kernels
    let mut build = cc::Build::new();

    build
        .cuda(true)
        .cudart("shared") // Link against shared CUDA runtime
        .include(&cuda_include)
        // RTX 4090 = SM89 (Ada Lovelace)
        // Also include SM80 for A100/3090 compatibility
        .flag("-gencode=arch=compute_80,code=sm_80")
        .flag("-gencode=arch=compute_89,code=sm_89")
        // Optimization flags
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-lineinfo") // For profiling
        // Suppress some warnings
        .flag("-Wno-deprecated-gpu-targets")
        // Enable FP16/BF16
        .flag("-DENABLE_FP16")
        .flag("-DENABLE_BF16");

    for cuda_file in &cuda_files {
        if std::path::Path::new(cuda_file).exists() {
            build.file(cuda_file);
        } else {
            println!("cargo:warning=CUDA kernel file not found: {cuda_file}");
        }
    }

    // Only compile if we have files
    if cuda_files.iter().any(|f| std::path::Path::new(f).exists()) {
        build.compile("flash_attn_kernels");

        // Link CUDA libraries
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-lib=cudart");
    }
}
