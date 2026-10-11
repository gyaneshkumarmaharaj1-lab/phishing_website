import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("1. Loading dataset...")
df = pd.read_csv('phishing_site_urls.csv')

# Take a sample or use full data (using full data may take a few minutes)
# df = df.sample(100000, random_state=42) # Uncomment if you want faster training

print("2. Extracting features from URLs...")
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

# Convert features into dataframe
feature_list = [extract_features(u) for u in df['url']]
X = pd.DataFrame(feature_list)

# Map labels: 'good' -> 0, 'bad' -> 1
y = df['label'].apply(lambda x: 1 if x == 'bad' else 0)

print("3. Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Training Random Forest model (this may take a moment)...")
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("5. Evaluating model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("6. Saving trained model to 'phishing_model.pkl'...")
joblib.dump(model, 'phishing_model.pkl')
print("Training complete! Model saved successfully.")