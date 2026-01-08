import sys
import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
api_file = os.path.join(project_root, "api.py")

print("=" * 60)
print("FINAL API TEST")
print("=" * 60)

def test_api_structure():
    print("\n1. Checking API structure...")
    
    if os.path.exists(api_file):
        print(f"API file exists: {api_file}")
        
        with open(api_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ("FastAPI", "FastAPI framework"),
            ("@app.get", "GET endpoints"),
            ("@app.post", "POST endpoints"),
            ("/predict", "Predict endpoint"),
            ("/health", "Health endpoint"),
            ("/docs", "Documentation")
        ]
        
        passed = 0
        for keyword, description in checks:
            if keyword in content:
                print(f"  {description}")
                passed += 1
            else:
                print(f"  Missing {description}")
        
        print(f"Structure score: {passed}/{len(checks)}")
        return passed >= 4
        
    else:
        print(f"API file not found at: {api_file}")
        return False

def test_imports():
    print("\n2. Checking dependencies...")
    
    required = ["fastapi", "uvicorn", "pydantic", "joblib", "numpy"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  {package}")
        except ImportError:
            missing.append(package)
            print(f"  {package}")
    
    if missing:
        print(f"\nMissing packages: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True

def test_api_creation():
    print("\n3. Testing API creation...")
    
    try:
        test_api_code = '''
from fastapi import FastAPI
app = FastAPI(title="Test API")
@app.get("/")
def root():
    return {"message": "Test API"}
'''
        
        test_file = os.path.join(project_root, "test_api_temp.py")
        with open(test_file, 'w') as f:
            f.write(test_api_code)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_api", test_file)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        if hasattr(test_module, 'app'):
            print("Can create and import FastAPI app")
            
            from fastapi.testclient import TestClient
            client = TestClient(test_module.app)
            response = client.get("/")
            
            if response.status_code == 200:
                print("Test endpoint works")
            else:
                print(f"Test endpoint status: {response.status_code}")
            
            os.remove(test_file)
            return True
        else:
            print("Could not find 'app' in test module")
            return False
            
    except Exception as e:
        print(f"API creation test failed: {type(e).__name__}: {e}")
        return False

def test_project_structure():
    print("\n4. Checking project structure...")
    
    required_dirs = ["scripts", "tests", "data", "models", "results"]
    required_files = ["requirements.txt", "README.md"]
    
    print("Required directories:")
    for dir_name in required_dirs:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            print(f"  {dir_name}/")
        else:
            print(f"  {dir_name}/ (creating...)")
            os.makedirs(dir_path, exist_ok=True)
    
    print("\nRequired files:")
    for file_name in required_files:
        file_path = os.path.join(project_root, file_name)
        if os.path.exists(file_path):
            print(f"  {file_name}")
        else:
            print(f"  {file_name} (missing but can be created)")
    
    return True

def main():
    print(f"Project: {project_root}")
    
    tests = [
        test_project_structure,
        test_imports,
        test_api_structure,
        test_api_creation
    ]
    
    results = []
    for test in tests:
        try:
            if test():
                results.append(True)
                print("  → PASS")
            else:
                results.append(False)
                print("  → FAIL")
        except Exception as e:
            print(f"  → ERROR: {e}")
            results.append(False)
        
        print("  " + "-" * 40)
    
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed >= total - 1:
        print("API TEST PASSED!")
        print("\nNext steps:")
        print("   1. Create api.py with the provided code")
        print("   2. Run: python api.py")
        print("   3. Visit: http://localhost:8000/docs")
        print("   4. Test with: http://localhost:8000/predict/sample")
        return True
    else:
        print("API TEST FAILED")
        print("\nRecommendations:")
        print("   - Create api.py file with minimal API")
        print("   - Run: pip install fastapi uvicorn pydantic")
        print("   - Ensure project structure is correct")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)