import json 
import requests

API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJRRjVkcjJXQ3RhUjhHRFE2b29pV0hpIiwiaWF0IjoxNzMwMDgxMTA5LCJleHAiOjE3MzE1OTY0MDAsInR5cGUiOiJhcGlfa2V5In0.ZQkH4OeUKWiupEBMjHzQQRi97sVHZ-OU-j-ye_qpHz0'

predicted_prices = [152.17, 92.23, 92.23, 92.1,
                    92.23, 92.43, 101.06, 101.28, 
                    138.32, 139.24, 216.99, 135.22, 
                    0.0, 209.55, 213.08, 206.86, 
                    152.32, 155.17, 155.03, 154.78,
                    155.92, 155.03, 101.02, 155.92]

result = {
    'submit_result': predicted_prices
}

success = requests.post('https://research-api.solarkim.com/submissions/cmpt-2024',
                    data=json.dumps(result),
                    headers={
                        'Authorization': f'Bearer {API_KEY}'
                    }).json()
print(success) 
