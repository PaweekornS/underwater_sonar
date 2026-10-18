import argparse
from ultralytics import YOLO
import time
import os
import torch

BASE_DIR = "/project/lt200304-dipmt/paweekorn/sss_project"

def benchmark_model(config):
    CFG = config
    model_id = CFG["model_id"]
    total_epochs = 100
    step = 10
    
    dataset_type, dataset_name = CFG["data"].split("/")[-2].split("_")
    project = f"{BASE_DIR}/runs/bench_{dataset_name}/{model_id}/{dataset_type}"
    os.makedirs(project, exist_ok=True) # Ensure project dir exists
    
    # Path to the base pretrained model to start with
    current_weights = os.path.join(BASE_DIR, "YOLO/base", f"{model_id}.pt")
    
    # Define the single summary file path for this dataset run
    summary_txt_path = os.path.join(project, f"{model_id}_{dataset_type}_full_benchmark.txt")
    
    # Initialize the file with a header
    with open(summary_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"BENCHMARK REPORT: {model_id} on {CFG["data"].split("/")[-2]} dataset\n")
        f.write("="*80 + "\n\n")

    start_time = time.perf_counter()

    for epoch_limit in range(step, total_epochs + step, step):
        print(f"\n------ Training Stage: {epoch_limit} Epochs ------\n")
        
        model = YOLO(current_weights)
        
        run_name = "checkpoint"
        model.train(
            data=CFG["data"],
            epochs=step,  
            imgsz=CFG["imgsz"],
            batch=CFG["batch"],
            project=project,
            name=run_name,
            exist_ok=True,
            amp=False,
            plots=False,
            save=True,
            verbose=False
        )
        
        current_weights = os.path.join(project, run_name, "weights", "best.pt")

        metrics = model.val(
            data=CFG["data"],
            split='test',
            project=project,
            name=run_name,
            exist_ok=True,
            plots=False,
            save=False,
            verbose=False
        )

        # Append results to the single summary file
        elapsed_hours = (time.perf_counter() - start_time) / 3600
        
        with open(summary_txt_path, 'a') as f:
            f.write(f"\n[ STAGE: {epoch_limit} TOTAL EPOCHS ]\n")
            f.write(f"Cumulative Time: {elapsed_hours:.4f} hours\n")
            f.write("-" * 40 + "\n")
            
            header = f"{'Class':<15} {'Images':<10} {'Instances':<10} {'Box(P)':<10} {'R':<10} {'mAP50':<10} {'mAP50-95':<10}\n"
            f.write(header)
            
            mp, mr, map50, map95 = metrics.mean_results()
            f.write(f"{'all':<15} {metrics.nt_per_class.sum():<10.0f} {metrics.nt_per_class.sum():<10.0f} {mp:<10.3f} {mr:<10.3f} {map50:<10.3f} {map95:<10.3f}\n")
            
            for i, name in enumerate(metrics.names.values()):
                res = metrics.class_result(i)
                f.write(f"{name:<15} {metrics.nt_per_class.sum():<10.0f} {metrics.nt_per_class[i]:<10.0f} {res[0]:<10.3f} {res[1]:<10.3f} {res[2]:<10.3f} {res[3]:<10.3f}\n")
            
            # f.write("\n" + "="*30 + " End of Stage " + "="*30 + "\n")

        del model
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="yolo11n")
    args = parser.parse_args()
    
    CFG = {
        "model_id": args.config,
        "imgsz": 640,
        "batch": 32,
    }
    
    datasets = ["real_klcylider2", "combined_klcylider2", "real_mine-detection", "combined_mine-detection"]
    
    for dataset_type in datasets:
        CFG["data"] = os.path.join(BASE_DIR, "datasets", dataset_type, "data.yaml")
        benchmark_model(CFG)

if __name__ == "__main__":
    main()