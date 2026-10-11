import sys
import joblib
import pandas as pd
import re
from urllib.parse import urlparse

# Load the saved model
model = joblib.load('phishing_model.pkl')

def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['count_dots'] = url.count('.')
    features['count_hyphens'] = url.count('-')
    features['count_at'] = url.count('@')
    features['count_question'] = url.count('?')
    features['count_percent'] = url.count('%')
    
    ip_pattern = re.compile(r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])')
    features['has_ip'] = 1 if ip_pattern.search(url) else 0
    
    parsed_url = urlparse(url)
    features['is_https'] = 1 if parsed_url.scheme == 'https' else 0
    
    hostname = parsed_url.hostname or ''
    features['subdomain_count'] = hostname.count('.') - 1 if hostname.count('.') > 1 else 0
    
    return features

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <URL>")
        sys.exit(1)
        
    test_url = sys.argv[1]
    features = extract_features(test_url)
    X_input = pd.DataFrame([features])
    
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]
    
    status = "PHISHING / MALICIOUS" if prediction == 1 else "LEGITIMATE"
    print(f"\nURL: {test_url}")
    print(f"Result: {status}")
    print(f"Malicious Probability: {probability * 100:.2f}%\n")