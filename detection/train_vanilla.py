import argparse
from ultralytics import YOLO
import time
import os

BASE_DIR = "/project/lt200304-dipmt/paweekorn/sss_project"

def benchmark_model(config):
    CFG = config
    model_id = CFG["model_id"]
    epochs = CFG["epochs"]
    
    model_path = os.path.join(BASE_DIR, "YOLO/base", f"{model_id}.pt")
    model = YOLO(model_path)
    
    dataset_type, dataset_name = CFG["data"].split("/")[-2].split("_")
    project = f"{BASE_DIR}/runs/bench_{dataset_name}/{model_id}/{dataset_type}"
    run_name = f"{epochs}_epochs"

    # Train the model
    start_time = time.perf_counter()
    results = model.train(
        data=CFG["data"], epochs=epochs,
        imgsz=CFG["imgsz"], batch=CFG["batch"],
        project=project, name=run_name,
        exist_ok=True, verbose=False,
        val=False, save=False, plots=False,
        amp=False,
    )
    train_elapsed = time.perf_counter() - start_time  # total training time (seconds)
    train_time_hours = train_elapsed / 3600  # convert to hours

    metrics = model.val(
        data=CFG["data"], split='test',
        project=project, name=run_name,
        exist_ok=True, verbose=False,
        save=False, plots=False,
    )
    
    # 2. Define the save path
    save_dir = os.path.join(project, run_name)
    txt_path = os.path.join(
        save_dir, f"{model_id}_{epochs}epochs_{dataset_type}_summary.txt"
    )
    
    # 3. Create the formatted string
    with open(txt_path, 'w') as f:
        # First line as requested
        f.write(f"{model_id} {epochs} epochs summary\n")
        f.write(f"Train on {dataset_type} dataset\n")
        f.write(f"Total training time: {train_time_hours:.4f} hours\n")
        f.write("-" * 80 + "\n")
        
        # Header for the table
        header = f"{'Class':<15} {'Images':<10} {'Instances':<10} {'Box(P)':<10} {'R':<10} {'mAP50':<10} {'mAP50-95':<10}\n"
        f.write(header)
        
        # Log the 'all' (overall) metrics
        mp, mr, map50, map95 = metrics.mean_results()
        stats_all = f"{'all':<15} {metrics.nt_per_class.sum():<10.0f} {metrics.nt_per_class.sum():<10.0f} {mp:<10.3f} {mr:<10.3f} {map50:<10.3f} {map95:<10.3f}\n"
        f.write(stats_all)
        
        # Log per-class metrics
        for i, name in enumerate(metrics.names.values()):
            try:
                p = metrics.class_result(i)[0]
                r = metrics.class_result(i)[1]
                m50 = metrics.class_result(i)[2]
                m95 = metrics.class_result(i)[3]
                instances = metrics.nt_per_class[i]
                
                row = f"{name:<15} {metrics.nt_per_class.sum():<10.0f} {instances:<10.0f} {p:<10.3f} {r:<10.3f} {m50:<10.3f} {m95:<10.3f}\n"
                f.write(row)
            except:
                pass

# ============
# Execution
# ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="yolo11n")
    args = parser.parse_args()
    
    CFG = {
        "data": "data.yaml",
        "model_id": args.config,
        "epochs": -1,
        "imgsz": 640,
        "batch": 32,
    }
    
    # Start training
    print("🚀 Start benchmark", args.config)
    
    for dataset_type in ["real_klcylider2", "combined_klcylider2", "real_mine-detection", "combined_mine-detection"]:
        print("Training on", dataset_type, "\n")
        CFG["data"] = os.path.join(BASE_DIR, "datasets", dataset_type, "data.yaml")
        
        for epochs in [20, 50, 100]:
            CFG["epochs"] = epochs
            benchmark_model(CFG)
        print("\n", "="*20, "\n")
    

if __name__ == "__main__":
    main()