#!/usr/bin/env python3
"""
Complete CADENCE System Launcher
Runs the full pipeline: Data Processing → Model Training → Backend → Frontend
"""
import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime
import structlog

logger = structlog.get_logger()

class CADENCESystemLauncher:
    """Manages the complete CADENCE system pipeline"""
    
    def __init__(self):
        self.processes = {}
        self.current_step = 0
        self.total_steps = 6
        
    def log_step(self, step_name: str):
        """Log current step"""
        self.current_step += 1
        logger.info("=" * 80)
        logger.info(f"STEP {self.current_step}/{self.total_steps}: {step_name}")
        logger.info("=" * 80)
    
    def check_requirements(self) -> bool:
        """Check if all requirements are met"""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8+ required")
            return False
        
        # Check required files
        required_files = [
            "real_cadence_training.py",
            "real_model_training.py", 
            "cadence_backend.py",
            "frontend_integration.py",
            "requirements.txt"
        ]
        
        for file in required_files:
            if not Path(file).exists():
                logger.error(f"❌ Required file missing: {file}")
                return False
        
        # Check if frontend directory exists
        if not Path("frontend").exists():
            logger.error("❌ Frontend directory not found")
            return False
        
        logger.info("✅ All requirements met")
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        try:
            # Install requirements
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                return False
            
            # Install additional packages that might be missing
            additional_packages = [
                "datasets", "sentence-transformers", "umap-learn", 
                "hdbscan", "fastapi", "uvicorn", "scikit-learn"
            ]
            
            for package in additional_packages:
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], capture_output=True, timeout=60)
                except:
                    pass  # Ignore failures for optional packages
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency installation failed: {e}")
            return False
    
    def run_data_processing(self) -> bool:
        """Run data processing pipeline"""
        logger.info("🔄 Starting data processing...")
        logger.info("This will download and process real Amazon datasets...")
        logger.info("Expected time: 10-20 minutes")
        
        try:
            result = subprocess.run([
                sys.executable, "real_cadence_training.py"
            ], timeout=3600)  # 1 hour timeout
            
            if result.returncode != 0:
                logger.error("❌ Data processing failed")
                return False
            
            # Check if processed data exists
            processed_dir = Path("processed_data")
            required_files = [
                "queries.parquet",
                "products.parquet", 
                "vocabulary.pkl",
                "cluster_mappings.json"
            ]
            
            for file in required_files:
                if not (processed_dir / file).exists():
                    logger.error(f"❌ Missing processed file: {file}")
                    return False
            
            logger.info("✅ Data processing completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Data processing timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            return False
    
    def run_model_training(self) -> bool:
        """Run model training"""
        logger.info("🧠 Starting model training...")
        logger.info("Training Query LM and Catalog LM...")
        logger.info("Expected time: 15-30 minutes")
        
        try:
            result = subprocess.run([
                sys.executable, "real_model_training.py"
            ], timeout=7200)  # 2 hour timeout
            
            if result.returncode != 0:
                logger.error("❌ Model training failed")
                return False
            
            # Check if trained models exist
            models_dir = Path("trained_models")
            required_files = [
                "real_cadence.pt",
                "real_cadence_vocab.pkl",
                "real_cadence_config.json"
            ]
            
            for file in required_files:
                if not (models_dir / file).exists():
                    logger.error(f"❌ Missing trained model file: {file}")
                    return False
            
            logger.info("✅ Model training completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Model training timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False
    
    def setup_frontend(self) -> bool:
        """Setup frontend integration"""
        logger.info("🎨 Setting up frontend integration...")
        
        try:
            # Run frontend integration
            result = subprocess.run([
                sys.executable, "frontend_integration.py"
            ], timeout=300)
            
            if result.returncode != 0:
                logger.error("❌ Frontend integration failed")
                return False
            
            # Install frontend dependencies
            frontend_dir = Path("frontend")
            if (frontend_dir / "package.json").exists():
                logger.info("📦 Installing frontend dependencies...")
                
                result = subprocess.run([
                    "npm", "install"
                ], cwd=frontend_dir, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    logger.warning(f"⚠️  Frontend npm install issues: {result.stderr}")
                    # Don't fail completely, frontend might still work
                
            logger.info("✅ Frontend setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Frontend setup failed: {e}")
            return False
    
    def start_backend(self) -> bool:
        """Start the backend server"""
        logger.info("🚀 Starting CADENCE backend server...")
        
        try:
            # Start backend in background
            self.processes['backend'] = subprocess.Popen([
                sys.executable, "cadence_backend.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give backend time to start
            time.sleep(5)
            
            # Check if backend is running
            if self.processes['backend'].poll() is None:
                logger.info("✅ Backend server started successfully")
                logger.info("   URL: http://localhost:8000")
                logger.info("   API Docs: http://localhost:8000/docs")
                return True
            else:
                logger.error("❌ Backend server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self) -> bool:
        """Start the frontend development server"""
        logger.info("🎨 Starting React frontend server...")
        
        try:
            frontend_dir = Path("frontend")
            
            # Start frontend in background
            self.processes['frontend'] = subprocess.Popen([
                "npm", "start"
            ], cwd=frontend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give frontend time to start
            time.sleep(10)
            
            # Check if frontend is running
            if self.processes['frontend'].poll() is None:
                logger.info("✅ Frontend server started successfully")
                logger.info("   URL: http://localhost:3000")
                return True
            else:
                logger.error("❌ Frontend server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete CADENCE pipeline"""
        logger.info("🚀 STARTING COMPLETE CADENCE SYSTEM")
        logger.info("=" * 80)
        logger.info("This will:")
        logger.info("1. Process real Amazon QAC + Products datasets")
        logger.info("2. Train Query LM and Catalog LM models")
        logger.info("3. Setup and integrate frontend")
        logger.info("4. Start backend API server")
        logger.info("5. Start React frontend")
        logger.info("6. Launch complete working system")
        logger.info("")
        logger.info("⏱️  Total estimated time: 30-60 minutes")
        logger.info("💾 Storage required: ~2-3 GB")
        logger.info("🔧 Hardware: Optimized for RTX 3050 4GB + 16GB RAM")
        logger.info("")
        
        user_input = input("Continue with complete setup? (y/N): ")
        if user_input.lower() != 'y':
            logger.info("Setup cancelled by user")
            return False
        
        try:
            # Step 1: Check requirements
            self.log_step("System Requirements Check")
            if not self.check_requirements():
                return False
            
            # Step 2: Install dependencies
            self.log_step("Installing Dependencies")
            if not self.install_dependencies():
                return False
            
            # Step 3: Process data
            self.log_step("Processing Amazon Datasets")
            if not self.run_data_processing():
                return False
            
            # Step 4: Train models
            self.log_step("Training CADENCE Models")
            if not self.run_model_training():
                return False
            
            # Step 5: Setup frontend
            self.log_step("Setting Up Frontend Integration")
            if not self.setup_frontend():
                return False
            
            # Step 6: Start servers
            self.log_step("Starting Complete System")
            
            if not self.start_backend():
                return False
            
            if not self.start_frontend():
                return False
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\n❌ Setup interrupted by user")
            self.cleanup()
            return False
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            self.cleanup()
            return False
    
    def cleanup(self):
        """Clean up running processes"""
        logger.info("🧹 Cleaning up processes...")
        
        for name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"   Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
    
    def show_final_instructions(self):
        """Show final instructions to user"""
        logger.info("\n" + "🎉" * 30)
        logger.info("✅ CADENCE SYSTEM RUNNING SUCCESSFULLY!")
        logger.info("🎉" * 30)
        logger.info("")
        logger.info("🌐 ACCESS YOUR SYSTEM:")
        logger.info("   Frontend: http://localhost:3000")
        logger.info("   Backend API: http://localhost:8000")
        logger.info("   API Documentation: http://localhost:8000/docs")
        logger.info("")
        logger.info("🔧 SYSTEM FEATURES:")
        logger.info("   ✅ Real-time query autocomplete")
        logger.info("   ✅ AI-powered product search")
        logger.info("   ✅ Advanced clustering categories")
        logger.info("   ✅ Real Amazon QAC + Products data")
        logger.info("   ✅ Memory-optimized for RTX 3050")
        logger.info("")
        logger.info("🎯 TEST THE SYSTEM:")
        logger.info("   1. Open http://localhost:3000 in your browser")
        logger.info("   2. Start typing in the search box")
        logger.info("   3. Watch real-time autocomplete suggestions")
        logger.info("   4. Press Enter to see search results")
        logger.info("   5. Try different queries and categories")
        logger.info("")
        logger.info("⚠️  IMPORTANT:")
        logger.info("   • Keep this terminal open to maintain servers")
        logger.info("   • Press Ctrl+C to stop the system")
        logger.info("   • Backend logs will show API requests")
        logger.info("")
        
        try:
            # Monitor processes
            while True:
                time.sleep(5)
                
                # Check if processes are still running
                backend_running = (self.processes.get('backend') and 
                                 self.processes['backend'].poll() is None)
                frontend_running = (self.processes.get('frontend') and 
                                  self.processes['frontend'].poll() is None)
                
                if not backend_running:
                    logger.warning("⚠️  Backend server stopped")
                    break
                
                if not frontend_running:
                    logger.warning("⚠️  Frontend server stopped")
                    break
                    
        except KeyboardInterrupt:
            logger.info("\n👋 Shutting down CADENCE system...")
            self.cleanup()
            logger.info("✅ System stopped successfully")

def main():
    """Main function"""
    launcher = CADENCESystemLauncher()
    
    try:
        success = launcher.run_complete_pipeline()
        
        if success:
            launcher.show_final_instructions()
        else:
            logger.error("\n❌ CADENCE SYSTEM SETUP FAILED")
            logger.info("Check the error messages above and try again")
            launcher.cleanup()
            
    except Exception as e:
        logger.error(f"❌ System launcher failed: {e}")
        launcher.cleanup()

if __name__ == "__main__":
    main()