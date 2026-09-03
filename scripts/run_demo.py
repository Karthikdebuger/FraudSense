import sys
from pathlib import Path
import uvicorn

def main():
    # 1. Add project root to sys.path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

    # 2. Check if trained model exists
    model_path = project_root / "models" / "lgb_model.joblib"
    
    if not model_path.exists():
        # 3. If not, print instructions to run generate_data.py and train_model.py first
        print("Error: Trained model not found at models/lgb_model.joblib")
        print("Please run the following scripts first:")
        print("  1. python scripts/generate_data.py")
        print("  2. python scripts/train_model.py")
        sys.exit(1)
        
    # 7. Print a banner
    banner = """
=========================================
  ___                   _ ____                     
 | __| _ _ __ _ _  _ __| / ___| ___ _ _  ___ ___ 
 | _|| '_/ _` | || / _` |\___ \/ -_) ' \(_-</ -_)
 |_| |_| \__,_|\_,_\__,_||____/\___|_||_/__/\___|
                                                 
=========================================
API docs URL:  http://localhost:8000/docs
Dashboard URL: http://localhost:8000
Press Ctrl+C to stop
=========================================
"""
    print(banner)
    
    # 5. Import and run the FastAPI app from src.api.main
    from src.api.main import app
    
    # 4 & 6. Start the FastAPI server using uvicorn on host='0.0.0.0', port=8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 8. Proper __main__ guard
if __name__ == "__main__":
    main()
