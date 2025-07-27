# scripts/cleanup_project.py
import os
import shutil
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)

class ProjectCleanup:
    """
    Clean up unnecessary files from Athena project
    Keeps only production and testing pipeline files
    """
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent
        self.cleanup_log = []
        
    def get_essential_files(self) -> Dict[str, List[str]]:
        """Define essential files for production/testing pipeline"""
        return {
            'config_files': [
                'config/settings.py',
                'config/symbols.txt',
                'config/symbols_metadata.json',
                'config/.env.example'
            ],
            
            'core_source_files': [
                'src/ingestion/polygon_client.py',
                'src/processing/data_cleaner.py', 
                'src/processing/feature_engineer.py',
                'config/__init__.py',
                'src/__init__.py',
                'src/ingestion/__init__.py',
                'src/processing/__init__.py'
            ],
            
            'production_scripts': [
                'scripts/load_historical_data.py',
                'scripts/collect_live_data.py',
                'scripts/process_to_silver.py',
                'scripts/load_finviz_symbols.py',
                'scripts/cleanup_project.py'  # Keep this script
            ],
            
            'project_files': [
                'README.md',
                'requirements.txt',
                '.env',
                '.env.example',
                '.gitignore',
                'LICENSE'
            ],
            
            'data_directories': [
                'data/bronze/',
                'data/silver/',
                'data/finviz/',
                'logs/'
            ]
        }
    
    def get_files_to_remove(self) -> Dict[str, List[str]]:
        """Define files and directories to remove"""
        return {
            'unnecessary_scripts': [
                'scripts/generate_symbol_list.py',  # Replaced by Finviz loader
                'scripts/ingest_market_data.py',    # Never existed, but check
                'scripts/complete_options_system.py', # Development file
                'src/processing/silver_processor.py',  # Old version
                'src/data_ingestion/',              # Old directory structure
                'scripts/test_*.py',                # Test scripts
                'scripts/debug_*.py',               # Debug scripts
                'scripts/prototype_*.py',           # Prototype scripts
                'scripts/temp_*.py',                # Temporary scripts
                'scripts/old_*.py',                 # Old scripts
                'test_*.py',                        # Root level test scripts
                'inspect_*.py',                     # Inspection scripts
                'debug_*.py',                       # Debug scripts
                'config/test_*.py',                 # Config test files
                'config/polygon_config.py'         # Old config file
            ],
            
            'old_data_directories': [
                'data/symbols/',                    # Generated symbols, replaced by Finviz
                'data/temp/',                       # Temporary data
                'data/debug/',                      # Debug data
                'data/test/',                       # Test data
                'data/cache/',                      # Cache data
                'data/raw/',                        # Old raw data structure
                'data/processed/'                   # Old processed data structure
            ],
            
            'development_files': [
                '*.pyc',                           # Python compiled files
                '__pycache__/',                    # Python cache directories
                '.pytest_cache/',                  # Pytest cache
                '.coverage',                       # Coverage files
                'htmlcov/',                        # Coverage HTML
                '*.log',                          # Old log files in root
                'temp_*',                         # Temporary files
                'debug_*',                        # Debug files
                'test_output*',                   # Test output files
                'backup_*',                       # Backup files
                '*.bak',                          # Backup files
                '*.tmp',                          # Temporary files
                '.DS_Store',                      # macOS files
                'Thumbs.db'                       # Windows files
            ],
            
            'unused_notebooks': [
                '*.ipynb',                        # Jupyter notebooks (move to archive if needed)
                'notebooks/',                     # Notebook directory
                'experiments/',                   # Experiment directory
                'research/',                      # Research directory
                'analysis/'                       # Analysis directory
            ]
        }
    
    def scan_project(self) -> Dict:
        """Scan project for files to keep/remove"""
        essential_files = self.get_essential_files()
        files_to_remove = self.get_files_to_remove()
        
        scan_result = {
            'essential_found': [],
            'essential_missing': [],
            'files_to_remove': [],
            'unknown_files': []
        }
        
        # Check essential files
        for category, file_list in essential_files.items():
            for file_path in file_list:
                full_path = self.project_root / file_path
                if full_path.exists():
                    scan_result['essential_found'].append(str(full_path))
                else:
                    scan_result['essential_missing'].append(str(full_path))
        
        # Find files to remove
        for category, patterns in files_to_remove.items():
            for pattern in patterns:
                # Handle glob patterns
                if '*' in pattern or '?' in pattern:
                    matches = list(self.project_root.rglob(pattern))
                    for match in matches:
                        if match.exists():
                            scan_result['files_to_remove'].append(str(match))
                else:
                    full_path = self.project_root / pattern
                    if full_path.exists():
                        scan_result['files_to_remove'].append(str(full_path))
        
        # Find unknown Python files (not in essential list)
        essential_set = set()
        for file_list in essential_files.values():
            essential_set.update(file_list)
        
        for py_file in self.project_root.rglob("*.py"):
            rel_path = py_file.relative_to(self.project_root)
            if str(rel_path) not in essential_set and str(py_file) not in scan_result['files_to_remove']:
                scan_result['unknown_files'].append(str(py_file))
        
        return scan_result
    
    def remove_file_or_dir(self, path: Path) -> bool:
        """Remove a file or directory"""
        try:
            if path.is_file():
                if not self.dry_run:
                    path.unlink()
                self.cleanup_log.append(f"Removed file: {path}")
                return True
            elif path.is_dir():
                if not self.dry_run:
                    shutil.rmtree(path)
                self.cleanup_log.append(f"Removed directory: {path}")
                return True
            return False
        except Exception as e:
            self.cleanup_log.append(f"Error removing {path}: {e}")
            return False
    
    def create_backup(self, files_to_remove: List[str]) -> Path:
        """Create backup of files before deletion"""
        if self.dry_run:
            return None
        
        backup_dir = self.project_root / "cleanup_backup" / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in files_to_remove:
            source = Path(file_path)
            if source.exists() and source.is_file():
                try:
                    rel_path = source.relative_to(self.project_root)
                    backup_file = backup_dir / rel_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, backup_file)
                except Exception as e:
                    logger.warning(f"Could not backup {source}: {e}")
        
        logger.info(f"Backup created at: {backup_dir}")
        return backup_dir
    
    def cleanup_project(self) -> Dict:
        """Perform project cleanup"""
        logger.info(f"Starting project cleanup (dry_run={self.dry_run})")
        
        # Scan project
        scan_result = self.scan_project()
        
        # Create backup if not dry run
        backup_dir = None
        if not self.dry_run and scan_result['files_to_remove']:
            backup_dir = self.create_backup(scan_result['files_to_remove'])
        
        # Remove files
        removed_count = 0
        for file_path in scan_result['files_to_remove']:
            if self.remove_file_or_dir(Path(file_path)):
                removed_count += 1
        
        # Remove empty directories
        empty_dirs_removed = 0
        if not self.dry_run:
            for root, dirs, files in os.walk(self.project_root, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        if dir_path.is_dir() and not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            empty_dirs_removed += 1
                            self.cleanup_log.append(f"Removed empty directory: {dir_path}")
                    except Exception as e:
                        pass
        
        # Generate cleanup report
        cleanup_result = {
            'dry_run': self.dry_run,
            'timestamp': datetime.now().isoformat(),
            'essential_files_found': len(scan_result['essential_found']),
            'essential_files_missing': len(scan_result['essential_missing']),
            'files_removed': removed_count,
            'empty_dirs_removed': empty_dirs_removed,
            'backup_location': str(backup_dir) if backup_dir else None,
            'missing_essential_files': scan_result['essential_missing'],
            'unknown_files_found': scan_result['unknown_files'],
            'cleanup_log': self.cleanup_log
        }
        
        # Save cleanup report
        report_file = self.project_root / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if not self.dry_run:
            with open(report_file, 'w') as f:
                json.dump(cleanup_result, f, indent=2)
            logger.info(f"Cleanup report saved: {report_file}")
        
        return cleanup_result
    
    def print_cleanup_summary(self, result: Dict):
        """Print cleanup summary"""
        mode = "DRY RUN" if result['dry_run'] else "ACTUAL CLEANUP"
        
        print(f"\n{'=' * 60}")
        print(f"🧹 PROJECT CLEANUP SUMMARY ({mode})")
        print(f"{'=' * 60}")
        print(f"✅ Essential files found: {result['essential_files_found']}")
        print(f"❌ Essential files missing: {result['essential_files_missing']}")
        print(f"🗑️  Files removed: {result['files_removed']}")
        print(f"📁 Empty directories removed: {result['empty_dirs_removed']}")
        
        if result['backup_location']:
            print(f"💾 Backup created: {result['backup_location']}")
        
        if result['missing_essential_files']:
            print(f"\n⚠️  Missing essential files:")
            for file in result['missing_essential_files']:
                print(f"  • {file}")
        
        if result['unknown_files_found']:
            print(f"\n❓ Unknown Python files found (review manually):")
            for file in result['unknown_files_found'][:10]:  # Show first 10
                print(f"  • {file}")
            if len(result['unknown_files_found']) > 10:
                print(f"  ... and {len(result['unknown_files_found']) - 10} more")
        
        print(f"{'=' * 60}")
        
        if result['dry_run']:
            print("🔍 This was a DRY RUN - no files were actually removed")
            print("Run with --execute to perform actual cleanup")
        else:
            print("✅ Cleanup completed!")
        
        print(f"{'=' * 60}")


def main():
    """Main cleanup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up Athena project files')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually perform cleanup (default is dry run)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Perform cleanup
    cleaner = ProjectCleanup(dry_run=not args.execute)
    
    if not args.execute:
        logger.info("🔍 Running in DRY RUN mode - no files will be removed")
        logger.info("Use --execute flag to perform actual cleanup")
    else:
        logger.info("⚠️  Running in EXECUTE mode - files will be removed!")
        logger.info("Backup will be created automatically")
    
    try:
        result = cleaner.cleanup_project()
        cleaner.print_cleanup_summary(result)
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


if __name__ == "__main__":
    main()