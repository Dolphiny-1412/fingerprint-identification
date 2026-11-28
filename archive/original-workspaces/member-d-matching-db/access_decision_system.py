"""
Step 3: Access Decision Logic
Complete fingerprint authentication system with intelligent access control
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import sys

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'minutiae'))
from fingerprint_matching import verify_fingerprint, extract_fingerprint_features
from spoof_detection import SpoofDetectionSystem


class AccessDecisionSystem:
    """
    Comprehensive access decision system for fingerprint authentication
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.spoof_detector = SpoofDetectionSystem()
        self.access_log = []
        self.user_sessions = {}
        self.security_events = []
        
        # Setup logging
        self._setup_logging()
        
    def _default_config(self) -> Dict:
        """Default configuration for access decisions"""
        return {
            'authentication': {
                'min_similarity_threshold': 0.6,
                'high_confidence_threshold': 0.8,
                'max_attempts_per_session': 3,
                'session_timeout_minutes': 30,
                'lockout_duration_minutes': 15
            },
            'spoof_protection': {
                'spoof_risk_threshold': 0.7,
                'require_liveness_check': True,
                'block_high_risk_spoofs': True
            },
            'security_policies': {
                'require_mfa_high_risk': True,
                'log_all_attempts': True,
                'alert_on_suspicious_activity': True,
                'rate_limit_attempts': True
            },
            'access_levels': {
                'admin': {
                    'min_similarity': 0.8,
                    'require_mfa': True,
                    'max_spoof_risk': 0.3
                },
                'user': {
                    'min_similarity': 0.6,
                    'require_mfa': False,
                    'max_spoof_risk': 0.5
                },
                'guest': {
                    'min_similarity': 0.7,
                    'require_mfa': True,
                    'max_spoof_risk': 0.2
                }
            }
        }
    
    def _setup_logging(self):
        """Setup logging for access decisions"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"access_decisions_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def authenticate_user(self, 
                         query_image_path: str, 
                         database_path: str,
                         user_id: str = None,
                         access_level: str = "user",
                         session_id: str = None) -> Dict:
        """
        Complete authentication process with access decision
        
        Args:
            query_image_path: Path to the query fingerprint image
            database_path: Path to the fingerprint database
            user_id: Optional user ID for targeted search
            access_level: Required access level (admin, user, guest)
            session_id: Optional session ID for tracking
            
        Returns:
            Dict containing complete authentication results
        """
        start_time = time.time()
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        # Initialize result
        auth_result = {
            'session_id': session_id,
            'user_id': user_id,
            'access_level': access_level,
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': 0,
            'authentication_success': False,
            'access_granted': False,
            'confidence_level': 'low',
            'security_actions': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Step 1: Check session security
            session_check = self._check_session_security(session_id, user_id)
            if not session_check['valid']:
                auth_result['security_actions'].append('session_blocked')
                auth_result['warnings'].append(session_check['reason'])
                auth_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
                return auth_result
            
            # Step 2: Spoof Detection
            self.logger.info(f"Session {session_id}: Starting spoof detection")
            spoof_result = self.spoof_detector.detect_spoof(query_image_path)
            
            if spoof_result.get('spoof_detected', False):
                auth_result['security_actions'].append('spoof_detected')
                auth_result['warnings'].append(f"Spoof detected with {spoof_result['confidence']} confidence")
                self._log_security_event(session_id, 'spoof_detected', spoof_result)
                
                if self.config['spoof_protection']['block_high_risk_spoofs']:
                    auth_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
                    return auth_result
            
            # Step 3: Fingerprint Matching
            self.logger.info(f"Session {session_id}: Starting fingerprint matching")
            match_result = verify_fingerprint(query_image_path, database_path, 
                                            self.config['authentication']['min_similarity_threshold'])
            
            if not match_result['success']:
                auth_result['warnings'].append("Fingerprint matching failed")
                auth_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
                return auth_result
            
            # Step 4: Access Decision Logic
            access_decision = self._make_access_decision(
                match_result, spoof_result, access_level, session_id
            )
            
            # Step 5: Update result
            auth_result.update(access_decision)
            
            # Step 6: Log and track
            self._log_authentication_attempt(auth_result)
            self._update_session_tracking(session_id, auth_result)
            
        except Exception as e:
            self.logger.error(f"Authentication error in session {session_id}: {str(e)}")
            auth_result['warnings'].append(f"System error: {str(e)}")
            auth_result['security_actions'].append('system_error')
        
        auth_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
        return auth_result
    
    def _check_session_security(self, session_id: str, user_id: str = None) -> Dict:
        """Check session security and rate limiting"""
        current_time = datetime.now()
        
        # Check if session is locked out
        if session_id in self.user_sessions:
            session_data = self.user_sessions[session_id]
            
            # Check lockout period
            if session_data.get('locked_until'):
                if current_time < session_data['locked_until']:
                    remaining = session_data['locked_until'] - current_time
                    return {
                        'valid': False,
                        'reason': f"Session locked out. Try again in {int(remaining.total_seconds() / 60)} minutes"
                    }
                else:
                    # Clear lockout
                    session_data.pop('locked_until', None)
            
            # Check attempt limits
            attempts = session_data.get('attempts', 0)
            if attempts >= self.config['authentication']['max_attempts_per_session']:
                # Lock out session
                lockout_duration = timedelta(minutes=self.config['authentication']['lockout_duration_minutes'])
                session_data['locked_until'] = current_time + lockout_duration
                return {
                    'valid': False,
                    'reason': f"Too many failed attempts. Locked for {self.config['authentication']['lockout_duration_minutes']} minutes"
                }
        
        return {'valid': True, 'reason': None}
    
    def _make_access_decision(self, 
                             match_result: Dict, 
                             spoof_result: Dict, 
                             access_level: str,
                             session_id: str) -> Dict:
        """Make final access decision based on all factors"""
        
        # Get access level requirements
        level_config = self.config['access_levels'].get(access_level, self.config['access_levels']['user'])
        
        # Extract key metrics
        similarity_score = match_result.get('best_match', {}).get('similarity_score', 0)
        spoof_risk = spoof_result.get('overall_spoof_risk', 1.0)
        spoof_detected = spoof_result.get('spoof_detected', False)
        
        # Initialize decision
        decision = {
            'authentication_success': False,
            'access_granted': False,
            'confidence_level': 'low',
            'decision_factors': {
                'similarity_score': similarity_score,
                'spoof_risk': spoof_risk,
                'spoof_detected': spoof_detected,
                'access_level_required': access_level
            }
        }
        
        # Decision Logic
        
        # 1. Check if spoof is detected and blocking is enabled
        if spoof_detected and self.config['spoof_protection']['block_high_risk_spoofs']:
            decision['authentication_success'] = False
            decision['access_granted'] = False
            decision['confidence_level'] = 'blocked'
            decision['reason'] = 'Spoof detected - access blocked'
            return decision
        
        # 2. Check similarity threshold for access level
        if similarity_score < level_config['min_similarity']:
            decision['authentication_success'] = False
            decision['access_granted'] = False
            decision['confidence_level'] = 'low'
            decision['reason'] = f'Similarity score {similarity_score:.3f} below threshold {level_config["min_similarity"]}'
            return decision
        
        # 3. Check spoof risk threshold for access level
        if spoof_risk > level_config['max_spoof_risk']:
            decision['authentication_success'] = False
            decision['access_granted'] = False
            decision['confidence_level'] = 'suspicious'
            decision['reason'] = f'Spoof risk {spoof_risk:.3f} above threshold {level_config["max_spoof_risk"]}'
            return decision
        
        # 4. Determine confidence level
        if similarity_score >= self.config['authentication']['high_confidence_threshold']:
            confidence = 'high'
        elif similarity_score >= level_config['min_similarity'] + 0.1:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # 5. Check if MFA is required
        mfa_required = level_config['require_mfa'] or (
            spoof_risk > 0.5 and self.config['security_policies']['require_mfa_high_risk']
        )
        
        # 6. Final decision
        if confidence in ['high', 'medium']:
            decision['authentication_success'] = True
            decision['access_granted'] = not mfa_required  # Grant access if no MFA required
            decision['confidence_level'] = confidence
            decision['reason'] = f'Authentication successful with {confidence} confidence'
            
            if mfa_required:
                decision['security_actions'].append('mfa_required')
                decision['recommendations'].append('Multi-factor authentication required for access')
        else:
            decision['authentication_success'] = False
            decision['access_granted'] = False
            decision['confidence_level'] = 'low'
            decision['reason'] = 'Insufficient confidence for access'
        
        return decision
    
    def _log_authentication_attempt(self, auth_result: Dict):
        """Log authentication attempt"""
        if self.config['security_policies']['log_all_attempts']:
            self.access_log.append(auth_result)
            
            # Log to file
            log_entry = {
                'timestamp': auth_result['timestamp'],
                'session_id': auth_result['session_id'],
                'user_id': auth_result['user_id'],
                'access_level': auth_result['access_level'],
                'success': auth_result['authentication_success'],
                'access_granted': auth_result['access_granted'],
                'confidence': auth_result['confidence_level'],
                'processing_time_ms': auth_result['processing_time_ms'],
                'security_actions': auth_result['security_actions']
            }
            
            self.logger.info(f"Authentication attempt: {log_entry}")
    
    def _log_security_event(self, session_id: str, event_type: str, details: Dict):
        """Log security events"""
        security_event = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'event_type': event_type,
            'details': details
        }
        
        self.security_events.append(security_event)
        self.logger.warning(f"Security event: {event_type} in session {session_id}")
    
    def _update_session_tracking(self, session_id: str, auth_result: Dict):
        """Update session tracking"""
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = {
                'created_at': datetime.now(),
                'attempts': 0,
                'successful_attempts': 0,
                'failed_attempts': 0
            }
        
        session_data = self.user_sessions[session_id]
        session_data['last_attempt'] = datetime.now()
        session_data['attempts'] += 1
        
        if auth_result['authentication_success']:
            session_data['successful_attempts'] += 1
        else:
            session_data['failed_attempts'] += 1
    
    def get_access_statistics(self) -> Dict:
        """Get access statistics"""
        if not self.access_log:
            return {'message': 'No authentication attempts logged'}
        
        total_attempts = len(self.access_log)
        successful_auths = sum(1 for log in self.access_log if log['authentication_success'])
        granted_access = sum(1 for log in self.access_log if log['access_granted'])
        
        # Calculate average processing time
        avg_processing_time = np.mean([log['processing_time_ms'] for log in self.access_log])
        
        # Security events
        spoof_detections = sum(1 for log in self.access_log if 'spoof_detected' in log['security_actions'])
        mfa_requirements = sum(1 for log in self.access_log if 'mfa_required' in log['security_actions'])
        
        return {
            'total_attempts': total_attempts,
            'successful_authentications': successful_auths,
            'access_granted': granted_access,
            'success_rate': successful_auths / total_attempts if total_attempts > 0 else 0,
            'access_rate': granted_access / total_attempts if total_attempts > 0 else 0,
            'average_processing_time_ms': avg_processing_time,
            'security_events': {
                'spoof_detections': spoof_detections,
                'mfa_requirements': mfa_requirements,
                'total_security_events': len(self.security_events)
            },
            'session_statistics': {
                'active_sessions': len(self.user_sessions),
                'locked_sessions': sum(1 for s in self.user_sessions.values() if 'locked_until' in s)
            }
        }


def test_access_system(database_path: str, test_images: List[str]):
    """Test the access decision system"""
    print("🔐 Testing Access Decision System")
    print("=" * 60)
    
    # Initialize system
    access_system = AccessDecisionSystem()
    
    # Test different access levels
    access_levels = ['guest', 'user', 'admin']
    
    for i, image_path in enumerate(test_images[:3]):  # Test first 3 images
        print(f"\n🧪 Test {i+1}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        for access_level in access_levels:
            print(f"\n🔑 Testing {access_level.upper()} access level:")
            
            result = access_system.authenticate_user(
                query_image_path=image_path,
                database_path=database_path,
                user_id=f"test_user_{i+1}",
                access_level=access_level
            )
            
            # Display results
            status_emoji = "✅" if result['authentication_success'] else "❌"
            access_emoji = "🔓" if result['access_granted'] else "🔒"
            
            print(f"   {status_emoji} Authentication: {result['authentication_success']}")
            print(f"   {access_emoji} Access Granted: {result['access_granted']}")
            print(f"   🎯 Confidence: {result['confidence_level'].upper()}")
            print(f"   ⏱️  Processing Time: {result['processing_time_ms']}ms")
            
            if result['warnings']:
                print(f"   ⚠️  Warnings: {', '.join(result['warnings'])}")
            
            if result['security_actions']:
                print(f"   🛡️  Security Actions: {', '.join(result['security_actions'])}")
            
            if result['recommendations']:
                print(f"   💡 Recommendations: {', '.join(result['recommendations'])}")
    
    # Display statistics
    print(f"\n📊 System Statistics:")
    print("-" * 40)
    stats = access_system.get_access_statistics()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")


def main():
    """Main function to test the access decision system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Access Decision System')
    parser.add_argument('--database', required=True, help='Path to fingerprint database')
    parser.add_argument('--test-images', nargs='+', help='Test image paths')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample images')
    
    args = parser.parse_args()
    
    if args.demo:
        # Demo with sample images from the dataset
        test_images = [
            "archive/fingerprints/DB1_B/101_1.tif",
            "archive/fingerprints/DB1_B/102_1.tif",
            "archive/fingerprints/DB2_B/101_1.tif"
        ]
        
        # Check if images exist
        existing_images = [img for img in test_images if os.path.exists(img)]
        if not existing_images:
            print("❌ Demo images not found. Please ensure the dataset is available.")
            return
        
        test_access_system(args.database, existing_images)
        
    elif args.test_images:
        test_access_system(args.database, args.test_images)
    else:
        print("Please provide --test-images or use --demo for demonstration")


if __name__ == "__main__":
    main()
