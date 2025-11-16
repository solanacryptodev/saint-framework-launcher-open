use saint_framework_launcher_open_lib::inspect_model::inspect_model_from_dir;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           SAINT FRAMEWORK - MODEL INSPECTION UTILITY             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Inspect Gemma 270M model
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  INSPECTING: Gemma 270M Model");
    println!("═══════════════════════════════════════════════════════════════════\n");
    
    match inspect_model_from_dir("src-tauri/models/gemma-270M") {
        Ok(_) => println!("✅ Gemma 270M inspection completed successfully\n"),
        Err(e) => eprintln!("❌ Failed to inspect Gemma 270M: {}\n", e),
    }

    // Inspect Qwen3 0.6B model (if it exists)
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  INSPECTING: Qwen3 0.6B Model (if available)");
    println!("═══════════════════════════════════════════════════════════════════\n");
    
    match inspect_model_from_dir("src-tauri/models/qwen3-0.6B") {
        Ok(_) => println!("✅ Qwen3 0.6B inspection completed successfully\n"),
        Err(e) => eprintln!("⚠️  Qwen3 0.6B not available or failed: {}\n", e),
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  INSPECTION COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════");
}
