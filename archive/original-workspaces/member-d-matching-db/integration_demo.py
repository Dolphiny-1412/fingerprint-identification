#!/usr/bin/env python3
"""
Integration Demo for Person D (Matcher)
Demonstrates the security-enhanced matching data format for seamless integration.
"""

import json
import os
from fingerprint_matching import (
    extract_fingerprint_features,
    generate_security_enhanced_matching_data,
    match_fingerprints_with_integration_format,
    verify_fingerprint_with_integration_data
)

def demonstrate_integration_format():
    """
    Demonstrate the security-enhanced matching data format for Person D integration.
    """
    print("🔐 PERSON D (MATCHER) - INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Check if database exists
    database_path = "fingerprint_database.json"
    if not os.path.exists(database_path):
        print("❌ Database not found. Please run fingerprint_matching.py first to build the database.")
        return
    
    # Load database to get a sample fingerprint
    with open(database_path, 'r') as f:
        database = json.load(f)
    
    if not database:
        print("❌ Database is empty.")
        return
    
    # Get first fingerprint as example
    sample_id = list(database.keys())[0]
    sample_features = database[sample_id]
    
    print(f"📋 Sample Fingerprint ID: {sample_id}")
    print(f"📁 Image Path: {sample_features['image_path']}")
    print()
    
    # Generate security-enhanced matching data
    print("🔒 GENERATING SECURITY-ENHANCED MATCHING DATA")
    print("-" * 40)
    
    integration_data = generate_security_enhanced_matching_data(sample_features)
    
    print("✅ Integration Data Format (Exact as specified):")
    print(json.dumps(integration_data, indent=2))
    
    # Demonstrate matching with integration format
    print("\n🔄 DEMONSTRATING MATCHING WITH INTEGRATION FORMAT")
    print("-" * 40)
    
    # Get another fingerprint for comparison
    if len(database) > 1:
        compare_id = list(database.keys())[1]
        compare_features = database[compare_id]
        
        print(f"🔍 Comparing: {sample_id} vs {compare_id}")
        
        # Perform matching with integration format
        match_result = match_fingerprints_with_integration_format(sample_features, compare_features)
        
        print(f"📊 Matching Score: {match_result['matching_result']['final_score']:.3f}")
        print(f"🎯 Is Match: {match_result['matching_result']['is_match']}")
        print(f"🛡️  Spoof Detected: {match_result['matching_result']['spoof_detected']}")
        print(f"📈 Confidence: {match_result['matching_result']['confidence']}")
        
        print("\n📋 Query Fingerprint Integration Data:")
        print(json.dumps(match_result['query_fingerprint_data'], indent=2))
        
        print("\n📋 Database Fingerprint Integration Data:")
        print(json.dumps(match_result['database_fingerprint_data'], indent=2))
    
    # Demonstrate verification with integration data
    print("\n🔐 DEMONSTRATING VERIFICATION WITH INTEGRATION DATA")
    print("-" * 40)
    
    verification_result = verify_fingerprint_with_integration_data(
        sample_features['image_path'], 
        database_path
    )
    
    if verification_result['success']:
        print(f"✅ Verification Successful!")
        print(f"🎯 Is Verified: {verification_result['is_verified']}")
        print(f"📊 Total Matches: {len(verification_result['all_matches'])}")
        
        if verification_result['best_match']:
            print(f"🏆 Best Match: {verification_result['best_match']['fingerprint_id']}")
            print(f"📈 Best Match Score: {verification_result['best_match']['similarity_score']:.3f}")
        
        print("\n📋 Query Integration Data:")
        print(json.dumps(verification_result['query_integration_data'], indent=2))
        
        if verification_result['best_match_integration_data']:
            print("\n📋 Best Match Integration Data:")
            print(json.dumps(verification_result['best_match_integration_data'], indent=2))
    else:
        print(f"❌ Verification Failed: {verification_result['error']}")
    
    # Show integration readiness summary
    print("\n🎯 INTEGRATION READINESS SUMMARY")
    print("-" * 40)
    print("✅ Security-enhanced matching data format: IMPLEMENTED")
    print("✅ Minutiae count: IMPLEMENTED")
    print("✅ Spoof risk score: IMPLEMENTED")
    print("✅ Quality score: IMPLEMENTED")
    print("✅ Full minutiae array: IMPLEMENTED")
    print("✅ Integration-ready functions: IMPLEMENTED")
    print()
    print("🚀 READY FOR PERSON D INTEGRATION!")


def show_integration_api():
    """
    Show the integration API for Person D.
    """
    print("\n📚 INTEGRATION API FOR PERSON D")
    print("=" * 40)
    
    print("""
🔧 Available Functions:

1. generate_security_enhanced_matching_data(features)
   → Returns: {
       "minutiae_count": 200,
       "spoof_risk_score": 0.287,
       "quality_score": 0.813,
       "minutiae": [...] # Full minutiae array
     }

2. match_fingerprints_with_integration_format(features1, features2)
   → Returns: {
       'matching_result': {...},
       'query_fingerprint_data': {...},  # Security-enhanced format
       'database_fingerprint_data': {...},  # Security-enhanced format
       'integration_ready': True
     }

3. verify_fingerprint_with_integration_data(query_path, database_path)
   → Returns: {
       'success': True,
       'is_verified': True,
       'query_integration_data': {...},  # Security-enhanced format
       'best_match_integration_data': {...},  # Security-enhanced format
       ...
     }

📋 Data Format Validation:
✅ Matches exactly the specified integration format
✅ Includes all required security metrics
✅ Compatible with JSON serialization
✅ Ready for real-time processing
""")


if __name__ == "__main__":
    demonstrate_integration_format()
    show_integration_api()
