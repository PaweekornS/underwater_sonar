from ultralytics import YOLO
from models.register import patch_yolo_engine
import argparse
import os
import time

BASE_DIR = "/project/lt200304-dipmt/paweekorn/sss_project"    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True) # e.g., "champion.yaml"
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    patch_yolo_engine()
    
    # The system now handles any architecture you throw at it!
    yaml_path = f"{BASE_DIR}/YOLO/custom/{args.config}"
    
    # Start training
    print("\nStart Training", args.config)
    for dataset in ["combined_klcylider2", "combined_mine-detection"]:
        project = f"{BASE_DIR}/runs/custom/{args.config[:-5]}"
        run_name = dataset.split("_")[-1]
        
        print("Train on", run_name, "dataset\n")
        pretrained = f"{BASE_DIR}/YOLO/base/yolo11s.pt" if args.config == "11-SDC.yaml" else f"{BASE_DIR}/YOLO/base/yolo11m.pt"
        model = YOLO(yaml_path, task="detect").load(pretrained)
        
        start_time = time.perf_counter()
        results = model.train(
            data=f"{BASE_DIR}/datasets/{dataset}/data.yaml",
            epochs=args.epochs,
            patience=args.patience,
            batch=32,
            device=0,
            project=project,
            name=run_name,
            exist_ok=True,
            save=True,
            plots=True,
            verbose=False,
        )
        train_elapsed = time.perf_counter() - start_time  # total training time in seconds
        train_time_hours = train_elapsed / 3600  # convert to hours
        
        print("Validation on test dataset!\n")
        metrics = model.val(
            data=f"{BASE_DIR}/datasets/{dataset}/data.yaml", 
            split="test",
            project=project,
            name=os.path.join(run_name),
            exist_ok=True,
            save=True,
            plots=True,
        )
        
        # 2. Define the save path
        save_dir = os.path.join(project, run_name)
        txt_path = os.path.join(
            save_dir, f"{args.config[:-5]}_{run_name}_summary.txt"
        )
        
        # 3. Create the formatted string
        with open(txt_path, 'w') as f:
            # First line as requested
            f.write(f"{args.config[:-5]} summary\n")
            f.write(f"Train on {run_name} dataset\n")
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
                p = metrics.class_result(i)[0]
                r = metrics.class_result(i)[1]
                m50 = metrics.class_result(i)[2]
                m95 = metrics.class_result(i)[3]
                instances = metrics.nt_per_class[i]
                
                row = f"{name:<15} {metrics.nt_per_class.sum():<10.0f} {instances:<10.0f} {p:<10.3f} {r:<10.3f} {m50:<10.3f} {m95:<10.3f}\n"
                f.write(row)

if __name__ == "__main__":
    main()