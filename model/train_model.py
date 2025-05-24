import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Expanded log dataset
logs = [
    "System boot completed successfully",                   # SystemEvent
    "Unauthorized login attempt detected",                  # SecurityIncident
    "Disk space is almost full",                            # Warning
    "Unable to locate configuration file",                  # ApplicationError
    "Debug mode enabled for user session",                  # Debugging
    "New software update is available",                     # SystemEvent
    "Firewall has blocked suspicious activity",             # SecurityIncident
    "Kernel panic occurred during boot",                    # SystemFailure
    "Scheduled backup completed",                           # SystemEvent
    "Memory leak detected in module",                       # SystemFailure
    "User admin logged out",                                # Informational
    "Service crashed unexpectedly",                         # SystemFailure
    "SSL certificate expired",                              # SecurityIncident
    "Cache cleared successfully",                           # Informational
    "Connection timeout while accessing database",          # ApplicationError
    "Monitoring daemon restarted",                          # SystemEvent
    "System temperature exceeds threshold",                 # Warning
    "Invalid password entered multiple times",              # SecurityIncident
    "Reboot required after patch installation",             # SystemEvent
    "Developer tools enabled in production",                # Warning
]

labels = [
    "SystemEvent",
    "SecurityIncident",
    "Warning",
    "ApplicationError",
    "Debugging",
    "SystemEvent",
    "SecurityIncident",
    "SystemFailure",
    "SystemEvent",
    "SystemFailure",
    "Informational",
    "SystemFailure",
    "SecurityIncident",
    "Informational",
    "ApplicationError",
    "SystemEvent",
    "Warning",
    "SecurityIncident",
    "SystemEvent",
    "Warning",
]


# Train vectorizer and model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(logs)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# Save
joblib.dump((vectorizer, model), "model/classifier.pkl")

print("✅ Model trained and saved.")
