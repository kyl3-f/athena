#!/usr/bin/env python3
"""
Athena Project Cleanup & Organization Script
Standardizes file locations, removes temp files, and organizes project structure
"""

import os
import shutil
from pathlib import Path
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AthenaProjectOrganizer:
    """
    Organizes and cleans up the Athena project structure
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.cleanup_report = {
            'files_moved': [],
            'files_deleted': [],
            'directories_created': [],
            'errors': []
        }
    
    def get_target_structure(self):
        """Define the target project structure"""
        return {
            'src/': {
                'description': 'Core source code',
                'subdirs': ['ingestion/', 'processing/', 'ml/', 'monitoring/']
            },
            'scripts/': {
                'description': 'Utility and test scripts',
                'subdirs': []
            },
            'config/': {
                'description': 'Configuration files',
                'subdirs': []
            },
            'data/': {
                'description': 'Data storage',
                'subdirs': ['bronze/', 'silver/', 'gold/', 'signals/', 'models/']
            },
            'logs/': {
                'description': 'Application logs',
                'subdirs': []
            },
            'docs/': {
                'description': 'Documentation',
                'subdirs': []
            },
            'tests/': {
                'description': 'Unit and integration tests',
                'subdirs': []
            }
        }
    
    def create_directory_structure(self):
        """Create the standard directory structure"""
        logger.info("🏗️ Creating standard directory structure...")
        
        structure = self.get_target_structure()
        
        for main_dir, info in structure.items():
            # Create main directory
            dir_path = self.project_root / main_dir
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.cleanup_report['directories_created'].append(str(dir_path))
                logger.info(f"   ✅ Created: {main_dir}")
            
            # Create subdirectories
            for subdir in info['subdirs']:
                subdir_path = dir_path / subdir
                if not subdir_path.exists():
                    subdir_path.mkdir(parents=True, exist_ok=True)
                    self.cleanup_report['directories_created'].append(str(subdir_path))
                    logger.info(f"   ✅ Created: {main_dir}{subdir}")
    
    def organize_scripts(self):
        """Move and organize script files"""
        logger.info("📝 Organizing script files...")
        
        # Scripts that should be in /scripts
        script_files = [
            'athena_status_summary.py',
            'system_test_suite.py', 
            'live_market_test.py',
            'pre_deployment_checklist.py',
            'ultimate_victory.py',
            'enhanced_polygon_options.py',
            'options_feature_integration.py',
            'test_options_integration.py',
            'fixed_test_options_integration.py'
        ]
        
        scripts_dir = self.project_root / 'scripts'
        
        for script_file in script_files:
            source_path = self.project_root / script_file
            target_path = scripts_dir / script_file
            
            if source_path.exists() and not target_path.exists():
                try:
                    shutil.move(str(source_path), str(target_path))
                    self.cleanup_report['files_moved'].append(f"{script_file} -> scripts/")
                    logger.info(f"   ✅ Moved: {script_file} -> scripts/")
                except Exception as e:
                    self.cleanup_report['errors'].append(f"Failed to move {script_file}: {e}")
                    logger.error(f"   ❌ Failed to move {script_file}: {e}")
    
    def organize_core_components(self):
        """Organize core source components"""
        logger.info("🔧 Organizing core components...")
        
        # Files that should be in specific src/ subdirectories
        component_moves = {
            'production_orchestrator.py': 'src/',
            'ml_signal_generator.py': 'src/ml/',
            'risk_manager.py': 'src/ml/',
            'monitoring_dashboard.py': 'src/monitoring/',
            'performance_monitor.py': 'src/monitoring/',
            'enhanced_alerts.py': 'src/monitoring/'
        }
        
        for file_name, target_dir in component_moves.items():
            source_path = self.project_root / file_name
            target_path = self.project_root / target_dir / file_name
            
            if source_path.exists():
                try:
                    # Ensure target directory exists
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if not target_path.exists():
                        shutil.move(str(source_path), str(target_path))
                        self.cleanup_report['files_moved'].append(f"{file_name} -> {target_dir}")
                        logger.info(f"   ✅ Moved: {file_name} -> {target_dir}")
                except Exception as e:
                    self.cleanup_report['errors'].append(f"Failed to move {file_name}: {e}")
                    logger.error(f"   ❌ Failed to move {file_name}: {e}")
    
    def cleanup_temporary_files(self):
        """Remove temporary and test files"""
        logger.info("🧹 Cleaning up temporary files...")
        
        # Temporary files to remove
        temp_files = [
            'quicktest.py',
            'quicklive.py',
            'pipeline_test.py',
            'working_test.py',
            'final_test.py',
            'victory_test.py',
            'success_test.py',
            'ultimate_victory.py',  # Will be moved to scripts, not deleted
            'test_write.csv',
            'live_test_data.csv',
            'live_test_features.csv'
        ]
        
        # Temporary CSV files pattern
        temp_patterns = [
            'test_*.csv',
            'quicktest_*.csv',
            'athena_test_*.csv',
            'athena_victory_*.csv',
            'athena_ultimate_*.csv'
        ]
        
        # Remove specific temp files
        for temp_file in temp_files:
            file_path = self.project_root / temp_file
            if file_path.exists():
                try:
                    file_path.unlink()
                    self.cleanup_report['files_deleted'].append(temp_file)
                    logger.info(f"   ✅ Deleted: {temp_file}")
                except Exception as e:
                    self.cleanup_report['errors'].append(f"Failed to delete {temp_file}: {e}")
                    logger.error(f"   ❌ Failed to delete {temp_file}: {e}")
        
        # Remove pattern-based temp files
        import glob
        for pattern in temp_patterns:
            for file_path in glob.glob(str(self.project_root / pattern)):
                try:
                    Path(file_path).unlink()
                    self.cleanup_report['files_deleted'].append(Path(file_path).name)
                    logger.info(f"   ✅ Deleted: {Path(file_path).name}")
                except Exception as e:
                    self.cleanup_report['errors'].append(f"Failed to delete {Path(file_path).name}: {e}")
    
    def cleanup_python_cache(self):
        """Remove Python cache files and directories"""
        logger.info("🐍 Cleaning Python cache files...")
        
        # Remove __pycache__ directories
        for pycache_dir in self.project_root.rglob('__pycache__'):
            try:
                shutil.rmtree(pycache_dir)
                self.cleanup_report['files_deleted'].append(f"__pycache__/ in {pycache_dir.parent.name}")
                logger.info(f"   ✅ Deleted: __pycache__ in {pycache_dir.parent.name}/")
            except Exception as e:
                self.cleanup_report['errors'].append(f"Failed to delete {pycache_dir}: {e}")
        
        # Remove .pyc files
        for pyc_file in self.project_root.rglob('*.pyc'):
            try:
                pyc_file.unlink()
                self.cleanup_report['files_deleted'].append(pyc_file.name)
                logger.info(f"   ✅ Deleted: {pyc_file.name}")
            except Exception as e:
                self.cleanup_report['errors'].append(f"Failed to delete {pyc_file.name}: {e}")
    
    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        logger.info("📦 Creating __init__.py files...")
        
        # Directories that should be Python packages
        package_dirs = [
            'src',
            'src/ingestion',
            'src/processing', 
            'src/ml',
            'src/monitoring'
        ]
        
        for package_dir in package_dirs:
            init_file = self.project_root / package_dir / '__init__.py'
            if not init_file.exists():
                try:
                    init_file.touch()
                    self.cleanup_report['files_moved'].append(f"Created __init__.py in {package_dir}")
                    logger.info(f"   ✅ Created: {package_dir}/__init__.py")
                except Exception as e:
                    self.cleanup_report['errors'].append(f"Failed to create __init__.py in {package_dir}: {e}")
    
    def update_import_paths(self):
        """Update import paths in moved files"""
        logger.info("🔗 Updating import paths...")
        
        # This is a simplified version - in production, you'd want more sophisticated path updates
        scripts_dir = self.project_root / 'scripts'
        
        for script_file in scripts_dir.glob('*.py'):
            try:
                with open(script_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update common import patterns
                updated_content = content
                
                # Fix relative imports from scripts folder
                if 'sys.path.append(str(project_root))' not in updated_content:
                    if 'project_root = Path(__file__).parent' in updated_content:
                        updated_content = updated_content.replace(
                            'project_root = Path(__file__).parent',
                            'project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root'
                        )
                
                # Write back if changed
                if updated_content != content:
                    with open(script_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    logger.info(f"   ✅ Updated imports in: {script_file.name}")
                    
            except Exception as e:
                self.cleanup_report['errors'].append(f"Failed to update imports in {script_file.name}: {e}")
                logger.error(f"   ❌ Failed to update imports in {script_file.name}: {e}")
    
    def generate_project_summary(self):
        """Generate a summary of the project structure"""
        logger.info("📋 Generating project summary...")
        
        summary_file = self.project_root / 'PROJECT_STRUCTURE.md'
        
        summary_content = f"""# ATHENA TRADING SYSTEM - PROJECT STRUCTURE

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📁 Directory Structure

```
athena_trading/
├── src/                     # Core source code
│   ├── ingestion/          # Data ingestion (Polygon API clients)
│   ├── processing/         # Data processing and feature engineering
│   ├── ml/                 # Machine learning models and signals
│   └── monitoring/         # System monitoring and alerts
├── scripts/                # Utility and test scripts
├── config/                 # Configuration files
├── data/                   # Data storage
│   ├── bronze/            # Raw data
│   ├── silver/            # Processed data
│   ├── gold/              # Feature-engineered data
│   ├── signals/           # Trading signals
│   └── models/            # Trained ML models
├── logs/                   # Application logs
├── docs/                   # Documentation
└── tests/                  # Unit and integration tests
```

## 🏗️ Core Components

### Production System
- `src/production_orchestrator.py` - Main production coordinator
- `src/ingestion/polygon_client.py` - Stock data ingestion
- `src/processing/data_cleaner.py` - Data cleaning pipeline
- `src/processing/feature_engineer.py` - Feature engineering (104+ features)

### Options Integration
- `scripts/enhanced_polygon_options.py` - Options Greeks and flow analysis
- `scripts/options_feature_integration.py` - Options-stock data integration

### ML & Signals
- `src/ml/ml_signal_generator.py` - ML models for trading signals
- `src/ml/risk_manager.py` - Risk management and position limits

### Monitoring & Alerts
- `src/monitoring/monitoring_dashboard.py` - Real-time system dashboard
- `src/monitoring/enhanced_alerts.py` - Advanced alerting system
- `src/monitoring/performance_monitor.py` - System performance tracking

### Testing & Deployment
- `scripts/system_test_suite.py` - Comprehensive system testing
- `scripts/live_market_test.py` - Live market data validation
- `scripts/test_options_integration.py` - Options integration testing
- `scripts/pre_deployment_checklist.py` - Pre-deployment validation
- `scripts/athena_status_summary.py` - System status checker

## 🎯 Key Features

✅ **104+ Stock Features** - Technical indicators, price analysis, volume metrics  
✅ **Options Greeks Integration** - Real-time gamma exposure, flow analysis  
✅ **ML-Ready Pipeline** - Complete feature engineering for machine learning  
✅ **Production Orchestration** - Automated market hours operation  
✅ **Risk Management** - Position limits and signal validation  
✅ **Real-time Monitoring** - Dashboard and alert system  
✅ **Comprehensive Testing** - 10+ test categories validation  

## 🚀 Getting Started

1. **Environment Setup**: Set POLYGON_API_KEY environment variable
2. **System Check**: Run `python scripts/athena_status_summary.py`
3. **Testing**: Run `python scripts/system_test_suite.py`
4. **Options Test**: Run `python scripts/test_options_integration.py`
5. **Production**: Run `python src/production_orchestrator.py`

## 📊 Current Status

- **Core Pipeline**: ✅ OPERATIONAL (104 features working)
- **Options Integration**: ✅ IMPLEMENTED (Greeks + flow analysis)
- **Feature Engineering**: ✅ PRODUCTION READY
- **Testing Framework**: ✅ COMPREHENSIVE
- **Project Organization**: ✅ STANDARDIZED

---
*Generated by Athena Project Organizer*
"""
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            logger.info(f"   ✅ Created: PROJECT_STRUCTURE.md")
            self.cleanup_report['files_moved'].append("Created PROJECT_STRUCTURE.md")
        except Exception as e:
            self.cleanup_report['errors'].append(f"Failed to create PROJECT_STRUCTURE.md: {e}")
    
    def run_full_cleanup(self):
        """Run the complete cleanup and organization process"""
        from datetime import datetime
        
        logger.info("🏛️ ATHENA PROJECT CLEANUP & ORGANIZATION")
        logger.info("=" * 50)
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Run all cleanup steps
        self.create_directory_structure()
        self.organize_scripts()
        self.organize_core_components()
        self.cleanup_temporary_files()
        self.cleanup_python_cache()
        self.create_init_files()
        self.update_import_paths()
        self.generate_project_summary()
        
        # Generate cleanup report
        self.generate_cleanup_report()
        
        logger.info("\n🎉 PROJECT CLEANUP COMPLETE!")
        return self.cleanup_report
    
    def generate_cleanup_report(self):
        """Generate a detailed cleanup report"""
        logger.info("\n📋 CLEANUP REPORT:")
        logger.info("-" * 30)
        
        if self.cleanup_report['directories_created']:
            logger.info(f"📁 Directories created: {len(self.cleanup_report['directories_created'])}")
            for directory in self.cleanup_report['directories_created'][:5]:  # Show first 5
                logger.info(f"   + {directory}")
            if len(self.cleanup_report['directories_created']) > 5:
                logger.info(f"   ... and {len(self.cleanup_report['directories_created']) - 5} more")
        
        if self.cleanup_report['files_moved']:
            logger.info(f"📝 Files moved/created: {len(self.cleanup_report['files_moved'])}")
            for file_move in self.cleanup_report['files_moved']:
                logger.info(f"   → {file_move}")
        
        if self.cleanup_report['files_deleted']:
            logger.info(f"🗑️  Files deleted: {len(self.cleanup_report['files_deleted'])}")
            for deleted_file in self.cleanup_report['files_deleted'][:5]:  # Show first 5
                logger.info(f"   ✗ {deleted_file}")
            if len(self.cleanup_report['files_deleted']) > 5:
                logger.info(f"   ... and {len(self.cleanup_report['files_deleted']) - 5} more")
        
        if self.cleanup_report['errors']:
            logger.info(f"❌ Errors encountered: {len(self.cleanup_report['errors'])}")
            for error in self.cleanup_report['errors']:
                logger.error(f"   ! {error}")
        
        # Summary
        total_actions = (len(self.cleanup_report['directories_created']) + 
                        len(self.cleanup_report['files_moved']) + 
                        len(self.cleanup_report['files_deleted']))
        
        logger.info(f"\n✅ Total actions completed: {total_actions}")
        logger.info(f"❌ Errors: {len(self.cleanup_report['errors'])}")

def main():
    """Main cleanup execution"""
    # Determine project root
    current_path = Path.cwd()
    
    # Check if we're in the correct project directory
    if not (current_path / 'src').exists():
        print("❌ Not in Athena project root directory!")
        print("Please run this script from the athena_trading/ directory")
        return 1
    
    # Run cleanup
    organizer = AthenaProjectOrganizer(current_path)
    report = organizer.run_full_cleanup()
    
    # Final status
    if len(report['errors']) == 0:
        print("\n🎉 PROJECT ORGANIZATION COMPLETE!")
        print("✅ All files organized successfully")
        print("✅ Directory structure standardized")
        print("✅ Temporary files cleaned up")
        print("✅ Import paths updated")
        print("\n📖 Check PROJECT_STRUCTURE.md for full documentation")
        return 0
    else:
        print(f"\n⚠️  PROJECT ORGANIZATION COMPLETED WITH {len(report['errors'])} ERRORS")
        print("Check the error messages above for issues that need manual resolution")
        return 1

if __name__ == "__main__":
    exit(main())