"""
Complete Fingerprint Authentication System
Integrates all components: Matching, Spoof Detection, and Access Decision Logic
"""

import os
import json
import time
from datetime import datetime
import sys

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'minutiae'))
from fingerprint_matching import verify_fingerprint, test_matching_system
from spoof_detection import SpoofDetectionSystem
from access_decision_system import AccessDecisionSystem


class CompleteFingerprintSystem:
    """
    Complete fingerprint authentication system integrating all components
    """
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.matching_system = None  # Will be initialized when needed
        self.spoof_detector = SpoofDetectionSystem()
        self.access_system = AccessDecisionSystem()
        
        # System status
        self.system_status = {
            'database_loaded': False,
            'components_ready': False,
            'last_maintenance': None
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        print("🚀 Initializing Complete Fingerprint Authentication System")
        print("=" * 70)
        
        # Check database
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    database = json.load(f)
                self.system_status['database_loaded'] = True
                print(f"✅ Database loaded: {len(database)} fingerprint entries")
            except Exception as e:
                print(f"❌ Database loading failed: {e}")
                return
        else:
            print(f"❌ Database not found: {self.database_path}")
            print("💡 Please run fingerprint_matching.py first to build the database")
            return
        
        # Test components
        print("\n🔧 Testing System Components:")
        
        # Test spoof detection
        try:
            test_image = "archive/fingerprints/DB1_B/101_1.tif"
            if os.path.exists(test_image):
                spoof_result = self.spoof_detector.detect_spoof(test_image)
                print(f"✅ Spoof Detection: Ready (test score: {spoof_result.get('overall_spoof_risk', 0):.3f})")
            else:
                print("⚠️  Spoof Detection: Ready (no test image available)")
        except Exception as e:
            print(f"❌ Spoof Detection: Failed - {e}")
            return
        
        # Test access decision system
        try:
            self.access_system.get_access_statistics()
            print("✅ Access Decision System: Ready")
        except Exception as e:
            print(f"❌ Access Decision System: Failed - {e}")
            return
        
        self.system_status['components_ready'] = True
        self.system_status['last_maintenance'] = datetime.now().isoformat()
        
        print("\n🎉 System Initialization Complete!")
        print("=" * 70)
    
    def authenticate_fingerprint(self, 
                               image_path: str, 
                               user_id: str = None,
                               access_level: str = "user",
                               detailed_report: bool = True) -> dict:
        """
        Complete fingerprint authentication workflow
        
        Args:
            image_path: Path to fingerprint image
            user_id: Optional user ID
            access_level: Required access level
            detailed_report: Whether to include detailed analysis
            
        Returns:
            Complete authentication report
        """
        if not self.system_status['components_ready']:
            return {
                'success': False,
                'error': 'System not properly initialized'
            }
        
        print(f"\n🔐 Fingerprint Authentication Request")
        print(f"📁 Image: {os.path.basename(image_path)}")
        print(f"👤 User: {user_id or 'Unknown'}")
        print(f"🔑 Access Level: {access_level.upper()}")
        print("-" * 50)
        
        start_time = time.time()
        
        # Step 1: Spoof Detection
        print("🛡️  Step 1: Spoof Detection Analysis...")
        spoof_result = self.spoof_detector.detect_spoof(image_path)
        
        spoof_status = "🚨 BLOCKED" if spoof_result.get('spoof_detected') else "✅ CLEAR"
        print(f"   Status: {spoof_status}")
        print(f"   Risk Score: {spoof_result.get('overall_spoof_risk', 0):.3f}")
        
        # Step 2: Fingerprint Matching
        print("\n🔍 Step 2: Fingerprint Matching...")
        match_result = verify_fingerprint(image_path, self.database_path, threshold=0.5)
        
        if match_result['success']:
            similarity = match_result['best_match']['similarity_score']
            match_status = "✅ MATCH" if similarity >= 0.6 else "⚠️  WEAK MATCH"
            print(f"   Status: {match_status}")
            print(f"   Similarity Score: {similarity:.3f}")
            if match_result['best_match']:
                print(f"   Best Match: {match_result['best_match']['fingerprint_id']}")
        else:
            print("   Status: ❌ NO MATCH")
        
        # Step 3: Access Decision
        print("\n🔓 Step 3: Access Decision...")
        access_result = self.access_system.authenticate_user(
            query_image_path=image_path,
            database_path=self.database_path,
            user_id=user_id,
            access_level=access_level
        )
        
        # Final result
        final_status = "🔓 ACCESS GRANTED" if access_result['access_granted'] else "🔒 ACCESS DENIED"
        print(f"   Final Decision: {final_status}")
        print(f"   Confidence: {access_result['confidence_level'].upper()}")
        
        # Compile complete report
        complete_report = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'user_id': user_id,
            'access_level': access_level,
            'processing_time_ms': int((time.time() - start_time) * 1000),
            'success': access_result['access_granted'],
            'final_decision': final_status,
            'spoof_analysis': {
                'detected': spoof_result.get('spoof_detected', False),
                'risk_score': spoof_result.get('overall_spoof_risk', 0),
                'confidence': spoof_result.get('confidence', 'unknown')
            },
            'matching_analysis': {
                'success': match_result['success'],
                'similarity_score': match_result.get('best_match', {}).get('similarity_score', 0),
                'best_match_id': match_result.get('best_match', {}).get('fingerprint_id', None)
            },
            'access_decision': {
                'authentication_success': access_result['authentication_success'],
                'access_granted': access_result['access_granted'],
                'confidence_level': access_result['confidence_level'],
                'security_actions': access_result['security_actions'],
                'warnings': access_result['warnings'],
                'recommendations': access_result['recommendations']
            }
        }
        
        if detailed_report:
            complete_report['detailed_analysis'] = {
                'spoof_detection': spoof_result,
                'fingerprint_matching': match_result,
                'access_decision': access_result
            }
        
        # Display summary
        print(f"\n📊 Authentication Summary:")
        print("-" * 30)
        print(f"⏱️  Total Processing Time: {complete_report['processing_time_ms']}ms")
        print(f"🎯 Final Result: {final_status}")
        print(f"🛡️  Security Level: {access_result['confidence_level'].upper()}")
        
        if access_result['warnings']:
            print(f"⚠️  Warnings: {len(access_result['warnings'])}")
        
        if access_result['recommendations']:
            print(f"💡 Recommendations: {len(access_result['recommendations'])}")
        
        return complete_report
    
    def run_system_diagnostics(self) -> dict:
        """Run comprehensive system diagnostics"""
        print("\n🔧 Running System Diagnostics")
        print("=" * 50)
        
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.system_status,
            'component_tests': {},
            'recommendations': []
        }
        
        # Test database
        try:
            with open(self.database_path, 'r') as f:
                database = json.load(f)
            diagnostics['component_tests']['database'] = {
                'status': 'healthy',
                'entries': len(database),
                'size_mb': os.path.getsize(self.database_path) / (1024 * 1024)
            }
        except Exception as e:
            diagnostics['component_tests']['database'] = {
                'status': 'error',
                'error': str(e)
            }
            diagnostics['recommendations'].append('Fix database access issues')
        
        # Test spoof detection
        try:
            test_image = "archive/fingerprints/DB1_B/101_1.tif"
            if os.path.exists(test_image):
                spoof_result = self.spoof_detector.detect_spoof(test_image)
                diagnostics['component_tests']['spoof_detection'] = {
                    'status': 'healthy',
                    'test_score': spoof_result.get('overall_spoof_risk', 0)
                }
            else:
                diagnostics['component_tests']['spoof_detection'] = {
                    'status': 'warning',
                    'message': 'No test image available'
                }
        except Exception as e:
            diagnostics['component_tests']['spoof_detection'] = {
                'status': 'error',
                'error': str(e)
            }
            diagnostics['recommendations'].append('Check spoof detection configuration')
        
        # Test access decision system
        try:
            stats = self.access_system.get_access_statistics()
            diagnostics['component_tests']['access_decision'] = {
                'status': 'healthy',
                'statistics': stats
            }
        except Exception as e:
            diagnostics['component_tests']['access_decision'] = {
                'status': 'error',
                'error': str(e)
            }
            diagnostics['recommendations'].append('Verify access decision system')
        
        # Display results
        for component, test_result in diagnostics['component_tests'].items():
            status_emoji = "✅" if test_result['status'] == 'healthy' else "⚠️" if test_result['status'] == 'warning' else "❌"
            print(f"{status_emoji} {component.replace('_', ' ').title()}: {test_result['status']}")
            
            if 'error' in test_result:
                print(f"   Error: {test_result['error']}")
            elif component == 'database':
                print(f"   Entries: {test_result['entries']}")
                print(f"   Size: {test_result['size_mb']:.2f} MB")
        
        if diagnostics['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in diagnostics['recommendations']:
                print(f"   • {rec}")
        
        return diagnostics
    
    def get_system_statistics(self) -> dict:
        """Get comprehensive system statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.system_status,
            'access_statistics': self.access_system.get_access_statistics(),
            'database_info': {}
        }
        
        # Database statistics
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    database = json.load(f)
                
                stats['database_info'] = {
                    'total_entries': len(database),
                    'file_size_mb': os.path.getsize(self.database_path) / (1024 * 1024),
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(self.database_path)).isoformat()
                }
            except Exception as e:
                stats['database_info']['error'] = str(e)
        
        return stats


def main():
    """Main function to demonstrate the complete system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Fingerprint Authentication System')
    parser.add_argument('--database', default='fingerprint_database.json', help='Path to fingerprint database')
    parser.add_argument('--image', help='Test image path')
    parser.add_argument('--folder', '-f', help='Custom folder path to scan for images')
    parser.add_argument('--diagnostics', action='store_true', help='Run system diagnostics')
    parser.add_argument('--demo', action='store_true', help='Run demo authentication')
    parser.add_argument('--scan-folder', action='store_true', help='Scan a folder for images and test them')
    parser.add_argument('--recursive', '-r', action='store_true', default=True, 
                       help='Search subdirectories recursively (default: True)')
    parser.add_argument('--no-recursive', action='store_true', 
                       help='Disable recursive search (only immediate directory)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = CompleteFingerprintSystem(args.database)
    
    if not system.system_status['components_ready']:
        print("❌ System initialization failed. Please check the database and components.")
        return
    
    if args.diagnostics:
        system.run_system_diagnostics()
    
    elif args.scan_folder:
        # Scan a folder for images and test them
        if not args.folder:
            print("❌ Please specify a folder with --folder when using --scan-folder")
            return
        
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            return
        
        # Import the detection function
        from fingerprint_matching import detect_all_images_in_folder
        
        # Determine recursive search
        recursive = args.recursive and not args.no_recursive
        
        print(f"\n🔍 Scanning folder: {args.folder}")
        print(f"📁 Recursive search: {'Yes' if recursive else 'No'}")
        print("=" * 60)
        
        # Detect images
        detected_images = detect_all_images_in_folder(args.folder, recursive=recursive)
        
        if not detected_images:
            print("❌ No images found in the specified folder!")
            return
        
        print(f"\n🎯 Testing Authentication on {len(detected_images)} Images")
        print("=" * 60)
        
        # Test authentication on detected images
        for i, image_path in enumerate(detected_images[:5]):  # Test first 5 images
            print(f"\n🧪 Test {i+1}: {os.path.basename(image_path)}")
            print("-" * 40)
            
            # Test different access levels
            for access_level in ['user', 'admin']:
                print(f"\n🔑 Testing {access_level.upper()} access:")
                result = system.authenticate_fingerprint(
                    image_path=image_path,
                    user_id=f"scanned_user_{i+1}",
                    access_level=access_level,
                    detailed_report=False
                )
        
        # Show final statistics
        print(f"\n📈 Final System Statistics:")
        print("=" * 30)
        stats = system.get_system_statistics()
        
        print(f"📊 Access Statistics:")
        access_stats = stats['access_statistics']
        if 'message' not in access_stats:
            print(f"   Total Attempts: {access_stats['total_attempts']}")
            print(f"   Success Rate: {access_stats['success_rate']:.1%}")
            print(f"   Access Rate: {access_stats['access_rate']:.1%}")
            print(f"   Avg Processing Time: {access_stats['average_processing_time_ms']:.0f}ms")
        
        print(f"🗄️  Database Info:")
        db_info = stats['database_info']
        if 'total_entries' in db_info:
            print(f"   Total Entries: {db_info['total_entries']}")
            print(f"   File Size: {db_info['file_size_mb']:.2f} MB")
    
    elif args.demo:
        # Demo with sample images
        demo_images = [
            "archive/fingerprints/DB1_B/101_1.tif",
            "archive/fingerprints/DB1_B/102_1.tif",
            "archive/fingerprints/DB2_B/101_1.tif"
        ]
        
        existing_images = [img for img in demo_images if os.path.exists(img)]
        
        if not existing_images:
            print("❌ Demo images not found. Please ensure the dataset is available.")
            return
        
        print(f"\n🎯 Running Demo Authentication Tests")
        print("=" * 50)
        
        for i, image_path in enumerate(existing_images[:2]):  # Test first 2 images
            print(f"\n🧪 Demo Test {i+1}: {os.path.basename(image_path)}")
            print("=" * 40)
            
            # Test different access levels
            for access_level in ['user', 'admin']:
                print(f"\n🔑 Testing {access_level.upper()} access:")
                result = system.authenticate_fingerprint(
                    image_path=image_path,
                    user_id=f"demo_user_{i+1}",
                    access_level=access_level,
                    detailed_report=False
                )
        
        # Show final statistics
        print(f"\n📈 Final System Statistics:")
        print("=" * 30)
        stats = system.get_system_statistics()
        
        print(f"📊 Access Statistics:")
        access_stats = stats['access_statistics']
        if 'message' not in access_stats:
            print(f"   Total Attempts: {access_stats['total_attempts']}")
            print(f"   Success Rate: {access_stats['success_rate']:.1%}")
            print(f"   Access Rate: {access_stats['access_rate']:.1%}")
            print(f"   Avg Processing Time: {access_stats['average_processing_time_ms']:.0f}ms")
        
        print(f"🗄️  Database Info:")
        db_info = stats['database_info']
        if 'total_entries' in db_info:
            print(f"   Total Entries: {db_info['total_entries']}")
            print(f"   File Size: {db_info['file_size_mb']:.2f} MB")
    
    elif args.image:
        # Single image authentication
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        result = system.authenticate_fingerprint(
            image_path=args.image,
            user_id="test_user",
            access_level="user"
        )
    
    else:
        print("Complete Fingerprint Authentication System")
        print("=" * 50)
        print("Available commands:")
        print("  --diagnostics       Run system diagnostics")
        print("  --demo             Run demo authentication tests")
        print("  --image PATH       Authenticate specific image")
        print("  --scan-folder      Scan a folder for images and test them")
        print("  --folder PATH      Specify custom folder path")
        print("  --recursive        Search subdirectories (default)")
        print("  --no-recursive     Only search immediate directory")
        print("\nExamples:")
        print("  python complete_fingerprint_system.py --demo")
        print("  python complete_fingerprint_system.py --scan-folder --folder /path/to/images")
        print("  python complete_fingerprint_system.py --image /path/to/fingerprint.tif")


if __name__ == "__main__":
    main()
