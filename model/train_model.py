import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Expanded log dataset
logs = [
    "System boot completed successfully",                   # SYSTEM_NOTIFICATION
    "Unauthorized login attempt detected",                  # SECURITY_ALERT
    "Disk space is almost full",                            # WARNING
    "Unable to locate configuration file",                  # ERROR
    "Debug mode enabled for user session",                  # DEBUG
    "New software update is available",                     # SYSTEM_NOTIFICATION
    "Firewall has blocked suspicious activity",             # SECURITY_ALERT
    "Kernel panic occurred during boot",                    # SYSTEM_ERROR
    "Scheduled backup completed",                           # SYSTEM_NOTIFICATION
    "Memory leak detected in module",                       # SYSTEM_ERROR
    "User admin logged out",                                # INFO
    "Service crashed unexpectedly",                         # SYSTEM_ERROR
    "SSL certificate expired",                              # SECURITY_ALERT
    "Cache cleared successfully",                           # INFO
    "Connection timeout while accessing database",          # ERROR
    "Monitoring daemon restarted",                          # SYSTEM_NOTIFICATION
    "System temperature exceeds threshold",                 # WARNING
    "Invalid password entered multiple times",              # SECURITY_ALERT
    "Reboot required after patch installation",             # SYSTEM_NOTIFICATION
    "Developer tools enabled in production",                # WARNING
]

labels = [
    "SYSTEM_NOTIFICATION",
    "SECURITY_ALERT",
    "WARNING",
    "ERROR",
    "DEBUG",
    "SYSTEM_NOTIFICATION",
    "SECURITY_ALERT",
    "SYSTEM_ERROR",
    "SYSTEM_NOTIFICATION",
    "SYSTEM_ERROR",
    "INFO",
    "SYSTEM_ERROR",
    "SECURITY_ALERT",
    "INFO",
    "ERROR",
    "SYSTEM_NOTIFICATION",
    "WARNING",
    "SECURITY_ALERT",
    "SYSTEM_NOTIFICATION",
    "WARNING",
]

# Train vectorizer and model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(logs)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# Save
joblib.dump((vectorizer, model), "model/classifier.pkl")

print("✅ Model trained and saved.")
