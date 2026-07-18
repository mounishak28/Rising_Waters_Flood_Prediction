# IBM Cloud Deployment Guide

## Prerequisites

- IBM Cloud Account
- Python 3.10
- GitHub Account

## Steps

1. Login to IBM Cloud.
2. Create a Cloud Foundry application.
3. Push the Flask application using the IBM CLI.
4. Install dependencies from `requirements.txt`.
5. Upload `saved_models/flood_model.pkl`.
6. Set the startup command:

```bash
python app.py
```

7. Deploy the application.
8. Test the public URL.