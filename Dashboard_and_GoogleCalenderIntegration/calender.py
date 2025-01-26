import requests
import json

# API endpoint for booking events through Cal.com
api_endpoint = "https://api.cal.com/v2/bookings"

# Variables for the event details
event_start_time = "2025-04-09T05:30:00Z"
event_id = 1716999
user_name = "Tejaswi"
user_email = "b7574303@gmail.com"
user_timezone = "Asia/Kolkata"
preferred_lang = "en"
guest_email_list = ["b7574303@gmail.com"]

# Prepare the payload data for the API request
payload_data = json.dumps({
    "start": event_start_time,
    "eventTypeId": event_id,
    "attendee": {
        "name": user_name,
        "email": user_email,
        "timeZone": user_timezone,
        "language": preferred_lang
    },
    "guests": guest_email_list
})

# API Authentication and headers setup
api_access_token = 'cal_live_41404c05e9199871253d6931590e16a0'
request_headers = {
    'Authorization': f'Bearer {api_access_token}',
    'Content-Type': 'application/json',
    'cal-api-version': '2024-08-13'
}

# Making a POST request to create the booking
response = requests.post(api_endpoint, headers=request_headers, data=payload_data)
# Handle the response based on the status code
response_data = response.json()
status_message = "Event booking created successfully!" if response.status_code == 201 else "Failed to create booking."

# Print the outcome
print(status_message)
print(json.dumps(response_data, indent=4))
